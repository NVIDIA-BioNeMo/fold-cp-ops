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
"""The SM90 GEMM's epilogue layer: accumulator -> registers -> SMEM -> global.

Owns everything downstream of the accumulator: the register/SMEM retile, the optional C load, the
per-subtile visit hooks a mixin composes its terms into, and the TMA store.

**The store seam is the one comm-fusion extends.** ``build_D_copy_fn`` returns the closure that
moves a finished subtile out of SMEM, and it defaults to a plain TMA store to the local output. A
distributed kernel overrides exactly that to write into a peer's symmetric heap instead, and
inherits the rest of the epilogue unchanged -- which is the alternative to copying the kernel.

The ``epi_*`` methods here are DEFAULTS, deliberately trivial: ``epi_visit_subtile`` returns its
input, ``epi_setup_postact`` returns None, ``epi_convert_postact`` is the identity. They exist so
a bare ``GemmSm90`` is concrete and emits exactly one store. Three of them were missing upstream,
where every mixin happened to supply them, so the base class died on ``AttributeError`` several
frames into the epilogue.
"""

from typing import Tuple, Type, Callable, Optional, Union, Literal  # noqa: F401
import math  # noqa: F401

from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp, warpgroup  # noqa: F401
import cutlass.utils.hopper_helpers as sm90_utils  # noqa: F401
from cutlass import Int32, Float32, Float16, Boolean, const_expr  # noqa: F401
from cutlass.utils import LayoutEnum  # noqa: F401

import fold_cp_ops._internal.copy_utils as copy_utils
from fold_cp_ops._internal.rounding import RoundingMode  # noqa: F401

from dataclasses import dataclass

from fold_cp_ops._internal.runtime_params import ParamsBase


class EpiPartition:
    """The register/SMEM partitioning one epilogue invocation needs for one accumulator.

    Returned by :meth:`GemmSm90EpilogueMixin.epilogue_partition`. Rebuilt per work tile rather than
    hoisted, because it is derived from the accumulator being stored and a derivation may store
    SEVERAL per tile -- the dual-gated kernel runs this once per output-N sub-tile.

    A trace-time Python object, so it costs nothing at runtime and is never passed across a
    ``@cute.jit`` boundary (the DSL flattens arguments and cannot accept an object).

    Attributes:
        tiled_copy_r2s: The register-to-SMEM tiled copy for the output subtile.
        tRS_rD: The output register fragment the visit hooks write into.
        tRS_sD: Its SMEM destination.
        tRS_rAcc: The accumulator re-tiled to the epilogue's subtile shape -- a VIEW of ``acc``'s
            registers, not a copy, so writing ``acc`` afterwards is visible through it.
        load_acc_subtile: Closure pulling subtile ``i`` out of ``tRS_rAcc`` into ``tRS_rD``.
        tiled_copy_s2r: The SMEM-to-register copy for the optional C addend, else None.
        tRS_rC: C's register fragment in the output's layout, else None.
        tSR_rC: C's register fragment in the load's layout, else None.
        tSR_sC: C's SMEM source, else None.
    """

    __slots__ = (
        "tiled_copy_r2s",
        "tRS_rD",
        "tRS_sD",
        "tRS_rAcc",
        "load_acc_subtile",
        "tiled_copy_s2r",
        "tRS_rC",
        "tSR_rC",
        "tSR_sC",
    )

    def __init__(self, **fields):
        """Bind every slot by keyword.

        Args:
            **fields: One entry per name in ``__slots__``. Keyword-only: four of the nine are
                same-typed register fragments, so a positional form would let two be swapped with
                no diagnostic.

        Raises:
            TypeError: If a slot is missing.
            AttributeError: If a name is not a declared slot.
        """
        missing = set(self.__slots__) - set(fields)
        if missing:
            raise TypeError(f"EpiPartition missing field(s): {sorted(missing)}")
        for k, v in fields.items():
            setattr(self, k, v)


class GemmSm90EpilogueMixin:
    @dataclass
    class EpilogueArguments:
        """The base class's epilogue arguments: empty.

        ``GemmSm90`` on its own computes ``D = A @ B^T`` with no epilogue terms, so there is nothing
        to pass. A mixin replaces this with its own -- usually a NamedTuple, since ``__call__``
        annotates ``epilogue_args`` as ``tuple``; a bare ``GemmSm90`` is therefore invoked with
        ``()`` rather than with an instance of this dataclass.
        """

        pass

    EpilogueParams = ParamsBase

    def maybe_override_epi_tile(self, epi_tile):
        """Default: no override. GemmGatedMixin overrides this to force epi_tile_n = max(32, 2*G)
        for the contiguous-chunk (bifurcated) gated layout."""
        return epi_tile

    def make_epilogue_tma(self, mD, mC, epilogue_args):
        """Build the TMA atoms for the epilogue's output store and optional C load.

        The store's reduction op is chosen here: ``"add"`` when the epilogue arguments ask to
        accumulate into D, ``"store"`` otherwise. That is a property of the DESCRIPTOR, not of the
        epilogue code -- a mismatch overwrites where the caller asked to accumulate, silently.

        Args:
            mD: Output tensor, or None when the kernel produces only a post-activation. Must be
                16-byte aligned in pitch and base; TMA reports neither.
            mC: Addend, or None. Absent means the beta term is compiled away entirely.
            epilogue_args: The epilogue arguments. Read only for ``add_to_output``, via
                ``hasattr`` so a mixin that has no such field is not required to declare one.

        Returns:
            ``(tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c)``, each pair None when its
            tensor was.

        Note:
            Requires ``_setup_attributes`` to have run -- ``epi_tile`` and the staged epilogue SMEM
            layouts are read here.
        """
        tma_atom_d, tma_tensor_d = None, None
        if const_expr(mD is not None):
            tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
                mD,
                self.epi_smem_layout_staged,
                self.epi_tile,
                op_type="store"
                if not (hasattr(epilogue_args, "add_to_output") and epilogue_args.add_to_output)
                else "add",
            )
        tma_atom_c, tma_tensor_c = None, None
        if const_expr(mC is not None):
            tma_atom_c, tma_tensor_c = self._make_tma_epi_atoms_and_tensors(
                mC, self.epi_c_smem_layout_staged, self.epi_tile, op_type="load"
            )
        return tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_epi_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
        op_type: Literal["store", "load", "add"],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for storing D or loading C.

        :param tensor_d: Output tensor D
        :type tensor_d: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        assert op_type in ["load", "store", "add"]
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile)
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if op_type == "load"
            else cpasync.CopyBulkTensorTileS2GOp()
            if op_type == "store"
            else cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD)
        )
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            op, tensor_d, epi_smem_layout, d_cta_v_layout
        )
        return tma_atom_d, tma_tensor_d

    def epilogue_partition(self, tiled_mma, acc, sD, sC, tidx, has_C):
        """Partition the registers and SMEM one epilogue invocation stores through.

        Everything between "here is an accumulator" and "the epilogue loop can run": the r2s copy
        and its register/SMEM fragments, the accumulator re-tiled to the epilogue subtile shape, and
        -- when a C addend exists -- the mirror s2r partitioning for the load.

        **The seam.** A derivation that stores MORE THAN ONE accumulator per work tile calls this
        once per accumulator instead of overriding the role. That is exactly what the dual-gated
        kernel does: one pass per output-N sub-tile, each with its own coordinate and subtile
        offset. On ``main`` that requirement is why it forked ``kernel()``.

        Args:
            tiled_mma: The MMA atom, which sets the accumulator's fragment layout.
            acc: The accumulator to be stored. ``tRS_rAcc`` is a VIEW of its registers, so a hook
                that rewrites ``acc`` afterwards is seen by the epilogue.
            sD: The output staging tile in SMEM. May be None only when the kernel has no D, in
                which case the caller must not use the returned r2s fields.
            sC: The C staging tile, or None when ``has_C`` is False.
            tidx: Thread index within the warpgroup; selects this thread's slice of both copies.
            has_C: Whether a C addend exists. ``const_expr``, so the s2r half is pruned entirely
                when False rather than built and ignored.

        Returns:
            An :class:`EpiPartition`. Its four C fields are None when ``has_C`` is False.

        Note:
            ``d_dtype`` may legitimately be None for a post-activation-only kernel; bf16 stands in
            for the LAYOUT selection in that case, which is safe because nothing is stored through
            it -- but a caller that then stores D would be storing through the wrong layout.
        """
        d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
        tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
            tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
        )
        # (R2S, R2S_M, R2S_N, num_epi)
        tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
        load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
        if const_expr(has_C):
            tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
            )
        else:
            tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None
        return EpiPartition(
            tiled_copy_r2s=tiled_copy_r2s,
            tRS_rD=tRS_rD,
            tRS_sD=tRS_sD,
            tRS_rAcc=tRS_rAcc,
            load_acc_subtile=load_acc_subtile,
            tiled_copy_s2r=tiled_copy_s2r,
            tRS_rC=tRS_rC,
            tSR_rC=tSR_rC,
            tSR_sC=tSR_sC,
        )

    @cute.jit
    def epilogue(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_pipeline: cutlass.pipeline.PipelineAsync,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        epi_read_state: cutlass.pipeline.PipelineState,
        epi_producer_state: Optional[cutlass.pipeline.PipelineState],
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_t2r: Optional[cute.TiledCopy],  # Only for Sm100
        tiled_copy_r2s: cute.TiledCopy,
        tRS_sD: cute.Tensor,
        tiled_copy_s2r: Optional[cute.ThrCopy],
        tSR_rC: Optional[cute.Tensor],
        tSR_sC: Optional[cute.Tensor],
        copy_D: Optional[Callable],
        copy_C: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: Boolean,
        subtile_offset: int = 0,
        epi_gate3: cutlass.Constexpr = False,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        """Write one work tile's accumulator out: retile, combine, stage through SMEM, TMA to GMEM.

        The shared epilogue for every kernel in this family. It walks the accumulator in epilogue
        subtiles and, per subtile, calls the ``epi_*`` hooks -- which is where a subclass's terms,
        gates and alternate stores enter. The base class's hooks are no-ops, so what this produces
        by default is a plain converted store.

        Args:
            params: The traced epilogue params.
            epi_smem_tensors: The epilogue's SMEM tensors.
            epi_pipeline: Pipeline guarding the C load, when there is one.
            epi_store_pipeline: Pipeline guarding the SMEM -> GMEM store stages.
            epi_read_state: Consumer state for the C pipeline, advanced and returned.
            epi_producer_state: Producer state for the store pipeline, or None.
            acc: The accumulator for this work tile.
            epi_tile: The epilogue subtile shape.
            tiled_copy_C_atom: Copy atom for the C load.
            tiled_copy_r2s: Register-to-shared copy.
            tiled_copy_s2r: Shared-to-register copy for C.
            tiled_copy_t2r: Tensor-memory-to-register copy, or None on SM90.
            load_acc_subtile: Closure loading one accumulator subtile into registers.
            tRS_rD: Destination register fragment for the subtile.
            tRS_rC: C register fragment, or None.
            tRS_sD: SMEM staging tensor for D.
            bSG_sD: Partitioned SMEM source for the TMA store.
            bSG_gD: Partitioned GMEM destination.
            copy_D: Closure issuing the store, or None when a subclass stores elsewhere.
            tile_coord_mnkl: This work tile's coordinate.
            tile_idx: Linear work-tile index.
            tidx: Thread index within the CTA.
            is_tma_warp: Whether this warp issues the TMA store arrive.
            epilogue_barrier: The named barrier the epilogue warps rendezvous on.
            num_prev_subtiles: How many subtiles this CTA has already written, so a subclass's
                per-subtile state (a rounding stream, a peer offset) stays continuous across the
                multiple epilogue invocations a fused kernel makes per work tile.

        Returns:
            ``(epi_read_state, epi_producer_state)``, both advanced.
        """
        has_C = const_expr(tRS_rC is not None)
        has_D = const_expr(copy_D is not None)

        # Setup postact output (returns None for default epilogue, context tuple for Act).  epi_gate3
        # (constexpr) routes the dual-gated kernel's SECOND output-gate pass to its own mPostAct3 +
        # sigmoid; all other epilogues ignore it (default False).
        postact_ctx = self.epi_setup_postact(
            params,
            epi_smem_tensors,
            tiled_copy_r2s,
            tiled_copy_t2r,
            tile_coord_mnkl,
            tidx,
            epi_gate3=epi_gate3,
        )

        # gate3 (dual-gated TriMul output-gate) CTA-tile width.  Two designs share this hook:
        #   REGION-AWARE (staged, _gate3_full_width=True): the gate3 acc IS the full BLK_N-wide dual
        #     accumulator (a gate3 work-tile is a standard BLK_N GEMM against B_all's W3 rows), so the
        #     gate3 epilogue iterates the FULL (BLK_M, BLK_N) tile (sigmoid elementwise, no halving).
        #   NATIVE half-width (stagec/legacy, _gate3_full_width=False): the gate3 acc is BLK_N//2-wide
        #     (a separate narrow accumulator), so the gate3 epilogue iterates a (BLK_M, BLK_N//2) tile.
        if const_expr(epi_gate3 and not getattr(self, "_gate3_full_width", False)):
            epi_cta_tile_mn = (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1] // 2)
        else:
            epi_cta_tile_mn = self.cta_tile_shape_mnk[:2]
        epi_tile_shape = cute.zipped_divide(cute.make_layout(epi_cta_tile_mn), epi_tile).shape[1]
        # We iterate over epi tiles in the N dimension first before the M dimension
        epi_tile_layout = cute.make_ordered_layout(epi_tile_shape, order=(1, 0))
        epi_tile_num = cute.size(epi_tile_shape)
        # subtile_offset: when a single work-tile drives MULTIPLE epilogue invocations (e.g. the
        # dual-gated cluster path's n_per_cta>1 multi-accumulator loop), each invocation must use a
        # DISTINCT slice of the epilogue subtile counter so the epi-store SMEM buffer rotation +
        # named-barrier generation stay continuous across invocations (sharing the same
        # num_prev_subtiles makes the 2nd invocation reuse the 1st's buffer slots, desyncing the
        # is_tma_warp producer_acquire from the all-consumer epilogue_barrier -> bar.sync divergence).
        num_prev_subtiles = (
            tile_scheduler.num_tiles_executed * epi_tile_num * self._epi_subtile_mult
            + subtile_offset
        )

        epi_tensors = self.epi_begin(
            params,
            epi_smem_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            epilogue_barrier,
            tidx,
            tile_scheduler.num_tiles_executed,
        )

        if const_expr(copy_C is not None):
            for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
                gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx)
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # The global memory coordinate for the current epi tile
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from acc to D registers
            load_acc_subtile(tRS_rD, epi_idx)
            epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, gmem_coord)
            if const_expr(has_C):
                epi_pipeline.consumer_wait(epi_read_state)
                cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC)
                # Fence to make sure shared memory read is visible to TMA load
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    epi_pipeline.consumer_release(epi_read_state)
                epi_read_state.advance()
            if const_expr(copy_C is not None and epi_idx + self.epi_c_stage < epi_tile_num):
                gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()
            tRS_rPostAct = self.epi_visit_subtile(
                params, epi_loop_tensors, tRS_rD, tRS_rC, epi_gate3=epi_gate3
            )
            # Convert and store postact if this epilogue produces one
            if const_expr(postact_ctx is not None):
                tRS_rPostAct_out = self.epi_convert_postact(
                    tRS_rPostAct,
                    epi_loop_tensors["sr_seed"],
                    tidx,
                    tile_coord_mnkl,
                    num_prev_subtiles,
                    epi_idx,
                    epi_gate3=epi_gate3,
                )
            if is_tma_warp:
                epi_store_pipeline.producer_acquire()
            epilogue_barrier.arrive_and_wait()
            # Copy from D registers to shared memory
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            if const_expr(has_D):
                if const_expr(
                    self.rounding_mode == RoundingMode.RS
                    and self.acc_dtype == cutlass.Float32
                    and self.d_dtype == cutlass.BFloat16
                ):
                    seed = epi_loop_tensors["sr_seed"] + (
                        tile_coord_mnkl[0] * 65537
                        + tile_coord_mnkl[1] * 257
                        + tile_coord_mnkl[3] * 17
                        + (num_prev_subtiles + epi_idx) * 7
                    )
                    copy_utils.sr_cvt_copy(
                        tiled_copy_r2s,
                        tRS_rD,
                        tRS_sD[None, None, None, epi_buffer],
                        seed,
                        tidx,
                    )
                else:
                    copy_utils.cvt_copy(
                        tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer]
                    )
            # Copy postact from registers to shared memory
            if const_expr(postact_ctx is not None):
                tiled_copy_postact_r2s, tRS_sPostAct, copy_postact = postact_ctx
                cute.copy(
                    tiled_copy_postact_r2s,
                    tiled_copy_postact_r2s.retile(tRS_rPostAct_out),
                    tRS_sPostAct[None, None, None, epi_buffer],
                )
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_view_async_shared()
            epilogue_barrier.arrive_and_wait()
            # Copy from shared memory to global memory
            if is_tma_warp:
                if const_expr(has_D):
                    copy_D(src_idx=epi_buffer, dst_idx=gmem_coord)
                if const_expr(postact_ctx is not None):
                    copy_postact(src_idx=epi_buffer, dst_idx=gmem_coord)
                epi_store_pipeline.producer_commit()

        self.epi_end(
            params,
            epi_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            tidx,
        )

        return epi_read_state, epi_producer_state

    def epi_retile_acc(self, acc, tRS_rD, tiled_copy_r2s):
        """Retile accumulator for epilogue subtile access. SM90 uses flat_divide."""
        return cute.flat_divide(acc, tRS_rD.layout)

    @cute.jit
    def epi_load_acc_subtile(self, tRS_rAcc: cute.Tensor, tRS_rD: cute.Tensor, epi_idx: int):
        """Copy one epilogue subtile out of the accumulator fragment into ``tRS_rD``.

        Args:
            tRS_rAcc: The retiled accumulator, subtile-indexed on its trailing mode.
            tRS_rD: Destination register fragment, overwritten.
            epi_idx: Which subtile to load.

        Returns:
            None. Overridable: a kernel accumulating in a different layout replaces this rather
            than the whole epilogue.
        """
        cute.autovec_copy(tRS_rAcc[None, None, None, epi_idx], tRS_rD)

    @cute.jit
    def epi_begin(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tidx: Int32,
        tile_idx=None,
    ) -> Tuple[cute.Tensor, ...]:
        """Per-work-tile epilogue setup hook. No-op on the base class.

        Args:
            params: The traced epilogue params.
            epi_smem_tensors: The epilogue's SMEM tensors.
            epi_tile: The epilogue subtile shape.
            tiled_copy_t2r: Tensor-memory-to-register copy, or None on SM90.
            tiled_copy_r2s: Register-to-shared copy.
            tile_coord_mnkl: This work tile's coordinate.
            epilogue_barrier: The epilogue's named barrier.
            tidx: Thread index within the CTA.
            tile_idx: Optional linear work-tile index.

        Returns:
            An empty tuple. A mixin returns its per-tile loaded values here.
        """
        return ()

    def epi_begin_loop(
        self, params: EpilogueParams, epi_tensors: Tuple[cute.Tensor, ...], epi_coord: cute.Coord
    ) -> Tuple[cute.Tensor, ...]:
        """Per-SUBTILE epilogue setup hook. No-op on the base class.

        Args:
            params: The traced epilogue params.
            epi_tensors: Whatever ``epi_begin`` returned.
            epi_coord: This subtile's coordinate within the work tile.

        Returns:
            An empty tuple. A mixin returns its per-subtile values here.
        """
        return ()

    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
        epi_gate3: cutlass.Constexpr = False,
    ) -> Optional[cute.Tensor]:
        """Combine one epilogue subtile. Identity on the base class.

        Args:
            params: The traced epilogue params.
            epi_loop_tensors: Whatever ``epi_begin_loop`` returned.
            tRS_rD: The subtile's register fragment, which a mixin modifies in place.
            tRS_rC: The C fragment, or None.
            epi_gate3: Whether this is a fused TriMul output-gate invocation.

        Returns:
            None -- no second (postact) output. A gated subclass returns the pre-activation
            fragment, which is what makes ``epilogue()`` build the second store.
        """
        return None

    def epi_visit_acc(
        self,
        params: EpilogueParams,
        acc: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tile_coord_mnkl: cute.Coord,
        tidx: Int32,
    ) -> None:
        """Hook to inspect or modify the WHOLE accumulator before it is subtiled. No-op here.

        Distinct from ``epi_visit_subtile``: that one sees a thread's slice of one subtile, this one
        sees the full accumulator with the MMA's own partitioning still intact -- which is what a
        cross-subtile reduction (a row max, a norm) needs.

        Args:
            params: The traced epilogue params.
            acc: The accumulator, modifiable in place.
            tiled_mma: The tiled MMA, for re-partitioning the accumulator.
            tile_coord_mnkl: This work tile's coordinate.
            tidx: Thread index within the CTA.

        Returns:
            None.
        """
        pass

    def epi_setup_postact(
        self,
        params: EpilogueParams,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        tidx,
        epi_gate3: cutlass.Constexpr = False,
    ):
        """Set up the SECOND epilogue output (the "postact" store), if this kernel has one.

        A gated GEMM writes two tensors per tile: the combined result, and the pre-activation it was
        combined from. `epilogue()` only builds that second store when this returns non-None, so the
        default of None is what makes a plain `D = A @ B` kernel emit exactly one store.

        This default exists so `GemmSm90` is **concrete**. It is the one epilogue hook upstream left
        undefined on the base -- every mixin happened to supply it -- which made instantiating the
        base directly die on `AttributeError: 'GemmSm90' object has no attribute
        'epi_setup_postact'`, several frames into `epilogue()`. Subclasses override it; nothing else
        changes for them, because an override is what they already had.

        Args:
            params: This kernel's epilogue params.
            epi_smem_tensors: The epilogue's SMEM tensors, as returned by `epi_get_smem_tensors`.
            tiled_copy_r2s: Register-to-shared copy for the epilogue subtile.
            tiled_copy_t2r: Tensor-memory-to-register copy, or None on SM90.
            tile_coord_mnkl: This work tile's coordinate.
            tidx: Thread index within the CTA.
            epi_gate3: Whether this invocation is the third (output-gate) epilogue of a fused
                TriMul. Ignored here.

        Returns:
            None -- no second output. An overriding subclass returns
            `(tiled_copy_postact_r2s, tRS_sPostAct, copy_postact)`, which `epilogue()` unpacks.
        """
        return None

    def epi_convert_postact(
        self,
        tRS_rPostAct,
        sr_seed,
        tidx,
        tile_coord_mnkl,
        num_prev_subtiles,
        epi_idx,
        epi_gate3: cutlass.Constexpr = False,
    ):
        """Convert the postact registers from the accumulator type to the store type.

        Unreachable unless `epi_setup_postact` returned non-None, so the identity default costs
        nothing in a kernel without a second output. Override to narrow, scale, or stochastically
        round the pre-activation on its way to SMEM.

        Args:
            tRS_rPostAct: The postact register fragment, as `epi_visit_subtile` returned it.
            sr_seed: Stochastic-rounding seed, or None under round-to-nearest.
            tidx: Thread index, mixed into the stochastic-rounding stream.
            tile_coord_mnkl: This work tile's coordinate.
            num_prev_subtiles: How many epilogue subtiles have already been written by this CTA,
                used to decorrelate the rounding stream across subtiles.
            epi_idx: Index of this subtile within the work tile.
            epi_gate3: Whether this is the output-gate epilogue of a fused TriMul. Ignored here.

        Returns:
            `tRS_rPostAct` unchanged.
        """
        return tRS_rPostAct

    @cute.jit
    def epi_end(
        self,
        params: EpilogueParams,
        epi_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        tidx,
    ) -> None:
        """Per-work-tile epilogue teardown hook. No-op on the base class.

        Args:
            params: The traced epilogue params.
            epi_tensors: Whatever ``epi_begin`` returned.
            epi_tile: The epilogue subtile shape.
            tiled_copy_t2r: Tensor-memory-to-register copy, or None on SM90.
            tiled_copy_r2s: Register-to-shared copy.
            tile_coord_mnkl: This work tile's coordinate.
            tidx: Thread index within the CTA.

        Returns:
            None.
        """
        pass

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        """Convert launch-time epilogue arguments into traced params. Identity on the base class.

        Args:
            args: The ``EpilogueArguments`` for this launch -- empty on the base class.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            An empty ``EpilogueParams``.
        """
        return self.EpilogueParams()

    def epi_get_tma_atoms(
        self, params: EpilogueParams, *, loc=None, ip=None
    ) -> list[cute.CopyAtom]:
        """Subclasses can override this"""
        return []

    @staticmethod
    def epi_smem_bytes_per_stage(
        args: Optional[EpilogueArguments],
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
    ) -> int:
        """SMEM bytes one epilogue pipeline stage needs. Zero on the base class.

        Args:
            args: The epilogue arguments, or None. Consulted so an absent term costs nothing.
            cta_tile_shape_mnk: The CTA tile, sizing any full-tile staging.
            epi_tile: The epilogue subtile, sizing any per-subtile staging.

        Returns:
            0. An underestimate by an override is a SMEM overrun at launch, not a tight fit.
        """
        return 0

    def epi_get_smem_struct(self, params: EpilogueParams):
        """The epilogue's SMEM struct type. Empty on the base class.

        Args:
            params: The traced epilogue params.

        Returns:
            A ``cute.struct`` with no members.
        """
        return cute.struct.MemRange[Int32, 0]  # Dummy struct

    def epi_get_smem_tensors(self, params: EpilogueParams, storage) -> Tuple[cute.Tensor, ...]:
        """The epilogue's SMEM tensors, sliced from the allocated struct. None on the base class.

        Args:
            params: The traced epilogue params.
            storage: The kernel's shared-storage object.

        Returns:
            An empty tuple.
        """
        return tuple()

    def epilog_smem_copy_atom(self, tiled_mma: cute.TiledMma) -> cute.TiledCopy:
        """Build the register-to-SMEM copy atom for the epilogue store.

        Picks the widest ``stmatrix``-class atom the accumulator type and MMA layout permit, falling
        back to a universal copy where none applies.

        Args:
            tiled_mma: The tiled MMA, whose accumulator layout the atom must match.

        Returns:
            A tiled copy for the register -> SMEM leg of the epilogue.
        """
        copy_atom_C = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(
                self.d_layout.is_m_major_c() if self.d_layout is not None else False,
                num_matrices=4 if self.epi_tile[1] % 16 == 0 else 2,
            ),
            Float16,  # this is just to get the right source layout
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        return tiled_copy_C_atom

    def epilog_smem_store_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        d_layout: Optional[LayoutEnum],
        dtype: Type[cutlass.Numeric],
        sD: Optional[cute.Tensor],
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Build the epilogue's SMEM store copy and partition D's staging tensor for this thread.

        Args:
            tiled_mma: The tiled MMA, whose accumulator partitioning the copy must match.
            d_layout: D's ``LayoutEnum``, or None when this kernel writes no D. It selects the
                store atom, so a wrong one stages the tile transposed.
            dtype: D's element type.
            sD: D's SMEM staging tensor, or None when there is no D.
            tidx: Thread index within the CTA.

        Returns:
            ``(tiled_copy_r2s, tRS_sD, tRS_rD)`` -- the copy, this thread's SMEM slice, and its
            register fragment.
        """
        if d_layout is None:
            d_layout = LayoutEnum.ROW_MAJOR
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        # Doesn't work with tile_N % 8 == 0 but tile_n % 16 != since this always
        # get st.matrix with num_matrices=4
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            d_layout, elem_ty_d=dtype, elem_ty_acc=self.acc_dtype
        )
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD) if sD is not None else None
        sD_shape = sD.shape[:2] if sD is not None else self.epi_tile
        tRS_rD_shape = thr_copy_r2s.partition_S(cute.make_identity_tensor(sD_shape)).shape
        tRS_rD = cute.make_rmem_tensor(tRS_rD_shape, self.acc_dtype)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_smem_load_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        c_layout: LayoutEnum,
        dtype: Type[cutlass.Numeric],
        sC: cute.Tensor,
        tRS_rD_layout: cutlass.Layout,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Build the epilogue's SMEM load copy for C and partition its staging tensor.

        The mirror of the store path, for the optional C addend: C arrives in SMEM by TMA and is
        read into registers with an ``ldmatrix``-class atom.

        Args:
            tiled_mma: The tiled MMA.
            c_layout: C's ``LayoutEnum``, which selects the (possibly transposing) load atom.
            dtype: C's element type.
            sC: C's SMEM staging tensor.
            tRS_rD_layout: The D fragment's layout, so C's fragment matches it element for element.
            tidx: Thread index within the CTA.

        Returns:
            ``(tiled_copy_s2r, tRS_sC, tRS_rC)``.
        """
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        copy_atom_s2r = copy_utils.sm90_get_smem_load_op(c_layout, dtype)
        tiled_copy_s2r = cute.make_tiled_copy_S(copy_atom_s2r, tiled_copy_C_atom)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_S(sC)
        tRS_rC = cute.make_rmem_tensor(tRS_rD_layout, dtype)
        tSR_rC = thr_copy_s2r.retile(tRS_rC)
        return tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC

    def epilog_gmem_copy_and_partition(
        self,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        mD_mn: cute.Tensor,
        tile_shape_mn: cute.Tile,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # (bM, bN)
        """Partition the GMEM output tile against its SMEM staging tile for the TMA store.

        Args:
            atom: The TMA store atom or tiled copy.
            mD_mn: The output tensor, already offset to this batch.
            tile_shape_mn: The CTA tile's ``(M, N)``.
            epi_tile: The epilogue subtile shape.
            sD: The SMEM staging tensor.
            tile_coord_mnkl: This work tile's coordinate, selecting which tile of ``mD_mn``.

        Returns:
            ``(bSG_sD, bSG_gD)`` -- the partitioned SMEM source and GMEM destination for the store.
        """
        gD = cute.local_tile(mD_mn, tile_shape_mn, tile_coord_mnkl[:2])
        tDgD_for_tma_partition = cute.zipped_divide(gD, epi_tile)
        is_s2g = isinstance(
            atom.op, (cpasync.CopyBulkTensorTileS2GOp, cpasync.CopyReduceBulkTensorTileS2GOp)
        )
        src_tensor, dst_tensor = (
            (sD, tDgD_for_tma_partition) if is_s2g else (tDgD_for_tma_partition, sD)
        )
        return copy_utils.tma_get_copy_fn(
            atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=src_tensor,
            dst_tensor=dst_tensor,
        )

    def build_D_copy_fn(
        self,
        tma_atom_d,
        mD_mnl: cute.Tensor,
        batch_idx: Int32,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
        epi_params,
        storage=None,
    ) -> Callable:
        """Build the D-store copy_fn (SMEM->GMEM TMA S2G partition for this CTA tile).

        Overridable store-build seam: the default returns the local TMA S2G copy_fn
        (the byte-identical local store). A subclass may override this to redirect the
        store (e.g. an in-epilogue peer-symmetric A2A store) using ``epi_params`` to
        carry any extra state across the @cute.jit -> @cute.kernel region. ``storage`` is
        the CTA ``SharedStorage`` (for a subclass that signals a consumer warpgroup via
        shared SMEM, e.g. the decoupled ring); the default store ignores it.
        """
        copy_D, _, _ = self.epilog_gmem_copy_and_partition(
            tma_atom_d,
            self.select_batch(mD_mnl, batch_idx),
            self.cta_tile_shape_mnk[:2],
            self.epi_tile,
            sD,
            tile_coord_mnkl,
        )
        return copy_D

    def make_epi_pipeline(
        self, c_smem_layout: cute.Layout | cute.ComposedLayout, epi_pipeline_mbar_ptr: cute.Pointer
    ):
        # Threads/warps participating in this pipeline
        """Construct the pipeline guarding the epilogue's C load.

        Args:
            c_smem_layout: C's staged SMEM layout, whose stage count sizes the pipeline and whose
                per-stage bytes set the TMA transaction count.
            epi_pipeline_mbar_ptr: SMEM pointer to the barrier array.

        Returns:
            The C-load pipeline.
        """
        epi_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Each warp will contribute 1 to the arrive count
        consumer_arrive_cnt = self.num_epi_warps
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        tma_copy_c_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=epi_pipeline_mbar_ptr,
            num_stages=self.epi_c_stage,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            tx_count=tma_copy_c_bytes,
            defer_sync=True,
        )

    def make_epi_store_pipeline(self):
        # Threads/warps participating in tma store pipeline
        """Construct the pipeline guarding the epilogue's SMEM -> GMEM store stages.

        Returns:
            A store pipeline whose stage count is the epilogue's, so the epilogue can reuse a
            staging buffer only once its previous TMA store has retired.
        """
        num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        epi_store_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_epi_threads)
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage, producer_group=epi_store_producer_group
        )
