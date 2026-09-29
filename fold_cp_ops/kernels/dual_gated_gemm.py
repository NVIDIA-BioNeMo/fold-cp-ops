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
"""The SM90 dual-gated GEMM: one 2N-wide GEMM whose epilogue folds it to N.

    gate[m, n] = (A @ Wg^T)[m, n] + bg[n]
    up  [m, n] = (A @ Wp^T)[m, n] + bp[n]
    out [m, n] = sigmoid(gate[m, n]) * up[m, n]   [* mask[m]]

This is the TriMul input projection with the LayerNorm taken off the front -- the caller supplies an
already-normalized activation. It is assembly only: the mainloop is `GemmSm90`'s and the epilogue is
`GemmGatedMixin`'s, so this module adds a class with no methods, a compile entry, and a front door.

**Where the LayerNorm fusions attach.** Two fusions form the same 2N pre-activation by different
means, and both plug into seams that already exist rather than forking this kernel:

* **Physical normalization** -- compute per-row statistics over A, normalize A in place in shared
  memory, then run the MMA on the normalized tile. It touches the MMA layer
  (`mma_setup_fragments` / `mma_initial_carry` / `mma_consume_work_tile`) and `_compute_stages` for
  the extra shared memory; the epilogue here is unchanged.
* **Algebraic correction** -- never materialize the normalized A. Accumulate the per-row statistics
  in the mainloop and apply the identity ``LN(x) @ B = r*(x @ Bw) - s*c + d`` as a rank-one
  correction in the epilogue. It overrides `GemmGatedMixin.epi_combine_preact` and nothing else.

Both then inherit the fold, the mask, the register permute and the store. That is the point of the
split: neither has to restate the epilogue, which is what the upstream versions of both did.

**The two weight layouts.** `chunk_g` selects how the 2N axis interleaves the gate and up
projections; `_internal/epi_gated.py` documents the register consequences. What matters at this
front door is the cost: ``chunk_g > 1`` loads `Wg` and `Wp` DIRECTLY through a two-tensor TMA and
needs no host preparation, while ``chunk_g == 1`` needs an element-interleaved ``(2N, K)`` weight,
which this entry builds per call -- in ONE fused launch that also emits the interleaved bias
(`interleave_dual_weights`), rather than the chain of torch ops that would cost four. Prefer
``chunk_g=16`` whenever the shapes allow it: one launch is still one more than none.
"""

from functools import partial
from typing import Callable, NamedTuple, Optional

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

import fold_cp_ops._internal.copy_utils as copy_utils
from fold_cp_ops._internal.activation import act_fn_map, as_gate_fn, gate_fn_map
from fold_cp_ops._internal.autotune import AxisSpace, TuneAxis, autotune
from fold_cp_ops._internal.arch import (
    check_arch_supported,
    get_max_active_clusters,
    require_sm90,
)
from fold_cp_ops._internal.cache_utils import jit_cache
from fold_cp_ops._internal.heuristic_arch import (
    TUNED_ARCH,
    heuristic_arch,
    warn_arch_suboptimal_once,
)
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.compile_time.layout_utils import transpose_view
from fold_cp_ops._internal.epi_default import GemmDefaultEpiMixin
from fold_cp_ops._internal.epi_gated import GemmGatedMixin, gated_epi_tile_fn
from fold_cp_ops._internal.epi_ops import (
    ChunkedHalfBiasLoad,
    Gate3RowVecLoad,
    TileStore,
    TransposedMaskColVecLoad,
)
from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
    check_broadcast_alignment,
    compile_gemm_kernel,
    div_for_dtype,
    get_major,
    make_fake_scheduler_args,
    make_scheduler_args,
    perm3d_single,
)
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops._internal.tensor_contract import check_tensor
from fold_cp_ops._internal.runtime_params import ParamsBase, mlir_namedtuple
from fold_cp_ops.kernels.gemm_sm90 import GemmSm90, GemmSm90Params

__all__ = [
    "append_gate3_weight",
    "build_dual_operands",
    "dual_gated_gemm",
    "DualGatedGemmSm90",
    "DualGatedGemmParams",
    "DualOperands",
    "interleave_dual_weights",
]


class DualGatedGemmParams(GemmSm90Params):
    """`GemmSm90`'s construction parameters plus the one the gate adds.

    Extending the base pack rather than declaring a second one is what keeps the guarantee total:
    `_bind_params` binds exactly one pack, so a subclass pack that did not include the base's fields
    would leave them unbound and the functor half-configured.

    Attributes:
        chunk_g: Contiguous up/gate chunk width along the 2N axis. Must be 1 (element interleave) or
            a multiple of 16 (block interleave); `GemmGatedMixin.maybe_override_epi_tile` asserts
            it. Chosen by the caller, read at trace time to select the register pairing and the
            epilogue tile, and immutable after construction -- upstream set it by patching the
            attribute onto the instance AFTER building it, which is the precise failure mode this
            mechanism exists to prevent: a value folded into a compiled kernel but reassignable
            afterwards, so the functor and the kernel can silently disagree.
        n_dual_tiles: Number of leading N work-tiles that belong to the DUAL projection, i.e.
            ``2N / tile_N``. Work tiles at or beyond it are output-gate tiles. 0 disables the
            output gate entirely and prunes its epilogue branch from the trace.
        gate3_n3: The output gate's true N extent, for predicating its partial last tile. 0 when
            there is no output gate. Must be 0 exactly when ``n_dual_tiles`` is 0.
    """

    chunk_g: int = 1
    n_dual_tiles: int = 0
    gate3_n3: int = 0


class DualGatedGemmSm90(GemmGatedMixin, GemmSm90):
    """`GemmSm90`'s mainloop under `GemmGatedMixin`'s fold, with `chunk_g` bound at construction.

    The base order is the content, as it is for `GemmDefaultSm90`: the epilogue mixin must precede
    `GemmSm90` so its `epi_*` hooks override the no-op defaults. `TemplateParamsMixin` is not listed
    -- `GemmSm90` already carries it, and naming it again here would make the MRO inconsistent.

    **The fused output gate lives here, not on a fusion.** ``out3 = act3(A @ W3^T + b3)`` needs
    nothing from any LayerNorm: it is a plain region of THIS GEMM against weight rows appended to
    the 2N operand, producing the same full-width accumulator the dual tiles produce. The only thing
    that differs is where the result goes. Three facts put it on this class rather than on a
    LayerNorm-fused derivation:

    * a caller wanting the gate WITHOUT a LayerNorm -- which the upstream cp=1 workflow does, for its
      separate-LN front -- could otherwise only get it by inheriting an LN kernel and switching the
      LN off, which is what the upstream does (a ``_normalize=False`` flag on the fused kernel);
    * the upstream implements the gate TWICE, once in each LN-fused variant, because neither can
      share it;
    * every mechanism it needs (`Gate3RowVecLoad`, the second `TileStore`, ``epi_gate3`` threading)
      already lives in the shared epilogue layer. Only the assembly was misplaced.

    Whether the gate is wanted is decided by where the LayerNorm SITS, not by which fusion is built:
    with the LayerNorm hoisted, the normalized activation is a real tensor and the gate can be a
    separate consumer; with it fused, the normalized activation exists only inside this kernel and
    the gate must ride along. That is a property of the caller, so the capability belongs to the
    class with no opinion about LayerNorm at all.

    Args:
        *args: Forwarded to `GemmSm90.__init__` -- ``(acc_dtype, ab_dtype, tile_shape_mn,
            cluster_shape_mnk)`` and its keyword configuration. See that constructor for the tile
            and cluster geometries it accepts and the ones it refuses.
        chunk_g: The weight layout, 1 or a multiple of 16. Bound as a compile-time parameter, so it
            is fixed for the life of the functor; a caller needing both layouts builds two functors.
            The kernel CANNOT verify that the weight actually has this layout -- it sees a
            ``(K, 2N)`` matrix either way -- so a mismatch is a wrong answer with no diagnostic.
        n_dual_tiles: Dual work-tile count, or 0 for no output gate.
        gate3_n3: The output gate's N extent, or 0.
        **kwargs: Forwarded to `GemmSm90.__init__`.

    Raises:
        TypeError: If `chunk_g` is not a compile-time value.
        RuntimeError: If the parameters are bound twice.
        ValueError: If ``n_dual_tiles`` and ``gate3_n3`` are not zero together, or propagated from
            `GemmSm90.__init__` for a tile geometry it refuses.
    """

    Params = DualGatedGemmParams

    #: The gated epilogue's ops, plus the fused output gate's two. Declaration ORDER defines the
    #: params struct and the shared-memory map, so the gate3 pair is INSERTED before the dual store
    #: rather than appended after it: an op moved is an op handed another op's buffer.
    #:
    #: The mask is `TransposedMaskColVecLoad` rather than the plain `ColVecLoad` `GemmGatedMixin`
    #: declares. For a 2-D ``(1, M)`` mask the two are byte-identical -- the transposed form falls
    #: through. It is declared here because an all-to-all-fused front hands a NATIVE 3-D mask, and
    #: having the op already in the params struct is what lets that be a launch-time change rather
    #: than a change to this class.
    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        TransposedMaskColVecLoad("mMaskColVec"),
        ChunkedHalfBiasLoad("mBiasUp"),
        ChunkedHalfBiasLoad("mBiasGate"),
        Gate3RowVecLoad("mRowVecBroadcast3"),
        TileStore("mPostAct3"),
        TileStore("mPostAct", epi_tile_fn=gated_epi_tile_fn),
    )
    #: ``act_fn`` is the dual gate (declared by `GemmGatedMixin`, redeclared because the field order
    #: defines the struct); ``act_fn_3`` is the output gate's activation.
    _extra_param_fields = (
        ("act_fn", cutlass.Constexpr, None),
        ("act_fn_3", cutlass.Constexpr, None),
    )
    _epi_param_bases = (ParamsBase,)

    #: The output gate's accumulator is the FULL tile_N-wide dual accumulator -- an output-gate work
    #: tile is a standard GEMM against the appended ``W3`` rows, not a narrow second pass. Read by
    #: the shared epilogue and by `Gate3RowVecLoad` to select full-width tiling.
    _gate3_full_width = True
    #: The output gate's own post-activation layout, always ``(M, N3)`` row-major. Latched per launch
    #: so ``transpose_out`` -- which makes the DUAL store m-major -- cannot poison the gate's store
    #: atom. None when there is no output gate.
    postact3_layout = None

    def __init__(self, *args, chunk_g: int = 1, n_dual_tiles: int = 0, gate3_n3: int = 0, **kwargs):
        """Bind `chunk_g` and the output-gate geometry in the base's single binding call.

        Args:
            *args: Positional arguments of `GemmSm90.__init__`.
            chunk_g: See the class docstring.
            n_dual_tiles: See the class docstring.
            gate3_n3: See the class docstring.
            **kwargs: Keyword arguments of `GemmSm90.__init__`.

        Returns:
            None.

        Raises:
            ValueError: If ``n_dual_tiles`` and ``gate3_n3`` are not zero together. They describe
                one feature from two sides, and a functor with a region boundary but no extent (or
                the reverse) would emit output-gate work tiles whose store has no destination.
        """
        if (n_dual_tiles == 0) != (gate3_n3 == 0):
            raise ValueError(
                f"n_dual_tiles and gate3_n3 describe one feature and must be zero together; got "
                f"n_dual_tiles={n_dual_tiles}, gate3_n3={gate3_n3}."
            )
        # Forwarded, not bound separately: `GemmSm90.__init__` makes the one and only
        # `_bind_params` call, so a subclass field has to travel INTO it. A second `_bind_params`
        # would raise, and rightly -- the pack is complete at construction or it is not a pack.
        super().__init__(
            *args, chunk_g=chunk_g, n_dual_tiles=n_dual_tiles, gate3_n3=gate3_n3, **kwargs
        )

    @property
    def _has_gate3(self) -> bool:
        """Whether the fused output gate is built. Compile-time; prunes its whole epilogue branch."""
        return self.n_dual_tiles > 0

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        """`GemmGatedMixin`'s epilogue terms plus the fused output gate's three.

        Attributes:
            mPostAct: ``(M, N)`` 16-bit gated output -- HALF the pre-activation width. n-major, or
                m-major for the transposed store.
            act_fn: The dual gate, a `Constexpr` callable taking ``(gate, up)``.
            alpha: Accumulator scale, or None.
            beta: C scale, or None.
            mRowVecBroadcast: ``(2N,)`` INTERLEAVED ``[bg, bp]`` bias. Its column order must match
                ``chunk_g``. Mutually exclusive with the two half-width vectors: both present
                double-count the bias, which nothing downstream can detect.
            mColVecBroadcast: ``(l, m)`` additive pre-activation bias, or None.
            mMaskColVec: ``(1, M)`` multiplicative POST-gate mask, or None.
            mBiasUp: ``(N,)`` up-projection bias for the block-interleave layout, or None.
            mBiasGate: ``(N,)`` gate-projection bias, likewise. Present exactly when `mBiasUp` is.
            mPostAct3: ``(M, N3)`` output-gate result, ALWAYS row-major, or None. Present exactly
                when the functor was built with a non-zero ``n_dual_tiles`` -- the output gate is a
                compile-time feature (it sets the scheduler's tile count), not a launch option.
            act_fn_3: The output gate's activation, a `Constexpr` callable of one argument. Present
                with `mPostAct3`.
            mRowVecBroadcast3: ``(N3,)`` output-gate bias added before its activation, or None.
            rounding_mode: `Constexpr` rounding; must be `RoundingMode.RN`.
            sr_seed: Stochastic-rounding seed; unused under RN.

        Note:
            The `Constexpr` fields are erased from the ABI, so at LAUNCH they must be passed None.
        """

        mPostAct: cute.Tensor
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

    def gated_params_dict(self, args):
        """`GemmGatedMixin`'s dict plus ``act_fn_3``, and the output gate's layout latched.

        Purpose
            The seam a further subclass extends -- the LayerNorm fusion adds three more fields to
            this dict. Keeping the output gate's own contribution HERE means that subclass does not
            have to know the gate exists.

        Semantics
            Side-effecting, like the base: it latches :attr:`postact3_layout` when an output gate is
            present. The gate's store is always row-major even when ``transpose_out`` has made the
            dual store m-major, so it carries its OWN layout rather than sharing one -- sharing it
            would pick a transposed ``stmatrix`` atom for a non-transposed tile.

        Args:
            args: This launch's :class:`EpilogueArguments`.

        Returns:
            The params dict, ready to be extended and splatted into ``self.EpilogueParams``.

        Raises:
            AssertionError: Propagated from the base for a refused geometry, or here if the presence
                of ``mPostAct3`` disagrees with how the functor was built.
        """
        assert (args.mPostAct3 is not None) == self._has_gate3, (
            f"mPostAct3 is {'present' if args.mPostAct3 is not None else 'absent'} but this functor "
            f"was built with n_dual_tiles={self.n_dual_tiles}. The output gate is a compile-time "
            f"feature: it decides the scheduler's tile count and whether its epilogue branch is "
            f"traced at all, so it cannot be turned on at launch."
        )
        d = super().gated_params_dict(args)
        if args.mPostAct3 is not None:
            self.postact3_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct3)
        d["act_fn_3"] = args.act_fn_3
        return d

    def epi_setup_postact(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        tidx,
        epi_gate3: cutlass.Constexpr = False,
    ):
        """Route the post-activation store to the dual output or to the output gate's own tensor.

        Purpose
            The two stores differ in three ways that all have to move together: the destination
            tensor, the tile WIDTH (the dual store is halved by the fold, the gate's is not) and the
            LAYOUT (the gate's is always row-major). Selecting them in one place is what stops a
            transposed dual store from picking the wrong ``stmatrix`` atom for the gate.

        Semantics
            Identical machinery to `GemmActMixin.epi_setup_postact`; only the three selections above
            are branched, all on the ``const_expr`` flag, so the dual path is unchanged.

        Args:
            params: This kernel's ``EpilogueParams``.
            epi_smem_tensors: The epilogue's SMEM tensors, indexed through ``self._epi_smem_map``.
            tiled_copy_r2s: D's register-to-SMEM tiled copy, the source layout this is built
                against so the two stay thread-consistent.
            tiled_copy_t2r: Unused on SM90; taken to match the base hook.
            tile_coord_mnkl: This work tile's coordinate. On the gate branch its ``[1]`` must
                already be the LOCAL gate n-block -- :meth:`run_epilogue` subtracts
                ``n_dual_tiles``.
            tidx: The calling thread's index.
            epi_gate3: Whether this is the output-gate pass.

        Returns:
            ``(tiled_copy_postact_r2s, tRS_sPostAct, copy_postact)``.
        """
        name = "mPostAct3" if const_expr(epi_gate3) else "mPostAct"
        sPostAct = epi_smem_tensors[self._epi_smem_map[name]]
        pa_layout = self.postact3_layout if const_expr(epi_gate3) else self.postact_layout
        copy_atom_postact_r2s = copy_utils.sm90_get_smem_store_atom(
            self.postact_dtype,
            transpose=pa_layout.is_m_major_c(),
            major_mode_size=getattr(params, f"epi_tile_{name}")[1],
        )
        tiled_copy_postact_r2s = cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_r2s)
        tRS_sPostAct = tiled_copy_postact_r2s.get_slice(tidx).partition_D(sPostAct)
        batch_idx = tile_coord_mnkl[3]
        cta_tile_postact_mn = (
            self.cta_tile_shape_mnk[:2] if const_expr(epi_gate3) else self.cta_tile_shape_postact_mn
        )
        copy_postact, _, _ = self.epilog_gmem_copy_and_partition(
            getattr(params, f"tma_atom_{name}"),
            self.select_batch(getattr(params, name), batch_idx),
            cta_tile_postact_mn,
            getattr(params, f"epi_tile_{name}"),
            sPostAct,
            tile_coord_mnkl,
        )
        return tiled_copy_postact_r2s, tRS_sPostAct, copy_postact

    @cute.jit
    def epi_visit_subtile(
        self, params, epi_loop_tensors, tRS_rD, tRS_rC=None, epi_gate3: cutlass.Constexpr = False
    ):
        """Fold the dual pre-activation, or apply the output gate elementwise.

        Semantics
            The dual pass is `GemmGatedMixin`'s, unchanged -- pre-activation, then the ``2N -> N``
            fold, then the mask.

            The output-gate pass is NOT a fold. Its accumulator is a full tile_N-wide GEMM against
            the appended ``W3`` rows, so the gate's bias is added to every column and the activation
            is applied ELEMENTWISE, one output per accumulator element. It carries no mask: the gate
            multiplies a downstream result, and masking it here would apply the mask twice.

        Args:
            params: This kernel's ``EpilogueParams``.
            epi_loop_tensors: This subtile's loaded terms, keyed by op name.
            tRS_rD: The accumulator fragment. On the dual pass it is updated in place to the
                pre-activation; on the gate pass it is read and left alone.
            tRS_rC: The C fragment, or None.
            epi_gate3: Whether this is the output-gate pass.

        Returns:
            The fp32 post-activation fragment -- half the input's size on the dual pass, the same
            size on the gate pass.
        """
        if const_expr(not epi_gate3):
            return GemmGatedMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        tDrRowVec3 = epi_loop_tensors["mRowVecBroadcast3"]
        if const_expr(tDrRowVec3 is not None):
            for i in cutlass.range(cute.size(tDrRowVec3), unroll_full=True):
                tRS_rD[i] += tDrRowVec3[i]
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
        for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
            tRS_rPostAct[i] = params.act_fn_3(tRS_rD[i])
        return tRS_rPostAct

    @cute.jit
    def run_epilogue(
        self,
        tiled_mma,
        acc,
        sD,
        sC,
        has_C: cutlass.Constexpr[bool],
        copy_D,
        copy_C,
        epilogue_params,
        epi_smem_tensors,
        epi_pipeline,
        epi_store_pipeline,
        epi_read_state,
        epi_producer_state,
        tile_coord_mnkl,
        tile_scheduler,
        tidx,
        is_tma_warp,
        epi_gate3: cutlass.Constexpr = False,
    ):
        """Send a work tile to the dual fold, or -- past the dual region -- to the output gate.

        Purpose
            Making the gate a REGION of the same operand -- ``W3``'s rows appended to the dual
            weight -- rather than a second pass means it costs no extra accumulator, no extra shared
            memory, and no second stream of work tiles. The only thing that differs is where the
            result goes, which is here.

        Semantics
            The region boundary is STATIC (``n_dual_tiles``, a compile-time count) and the test
            against it is a RUNTIME comparison on the work tile's N block, because a persistent CTA
            sees tiles from both regions. Both arms are therefore traced, and the branch selects
            one. When the gate is absent the whole comparison is ``const_expr``-pruned and the dual
            arm is all that exists -- the emitted kernel is the one without the feature.

            The gate arm re-bases the coordinate by subtracting ``n_dual_tiles``, so its store
            addresses ``mPostAct3``'s own origin; its TMA descriptor carries the true ``N3`` extent,
            which predicates the partial last tile.

        Args:
            See :meth:`GemmSm90MmaMixin.run_epilogue`. ``epi_gate3`` is accepted for signature
            compatibility and must be False: this override is what DECIDES the arm, so a caller
            forcing it would be overriding the region test.

        Returns:
            ``(epi_read_state, epi_producer_state)``.

        Raises:
            AssertionError: If ``epi_gate3`` is True on entry.
        """
        assert not epi_gate3, (
            "run_epilogue chooses the arm from the work tile's N block; passing epi_gate3=True "
            "would override that decision."
        )
        base = partial(
            GemmSm90.run_epilogue,
            self,
            tiled_mma,
            acc,
            sD,
            sC,
            has_C,
            copy_D,
            copy_C,
            epilogue_params,
            epi_smem_tensors,
            epi_pipeline,
            epi_store_pipeline,
        )
        if const_expr(not self._has_gate3):
            return base(
                epi_read_state,
                epi_producer_state,
                tile_coord_mnkl,
                tile_scheduler,
                tidx,
                is_tma_warp,
            )
        if tile_coord_mnkl[1] >= Int32(self.n_dual_tiles):
            epi_read_state, epi_producer_state = base(
                epi_read_state,
                epi_producer_state,
                (
                    tile_coord_mnkl[0],
                    tile_coord_mnkl[1] - Int32(self.n_dual_tiles),
                    tile_coord_mnkl[2],
                    tile_coord_mnkl[3],
                ),
                tile_scheduler,
                tidx,
                is_tma_warp,
                epi_gate3=True,
            )
        else:
            epi_read_state, epi_producer_state = base(
                epi_read_state,
                epi_producer_state,
                tile_coord_mnkl,
                tile_scheduler,
                tidx,
                is_tma_warp,
            )
        return epi_read_state, epi_producer_state


@jit_cache
def _compile_dual_gated_gemm(
    a_dtype,
    b_dtype,
    postact_dtype,
    a_major,
    b_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    activation,
    rowvec_dtype,
    maskvec_dtype,
    half_bias_dtype,
    gate3_activation,
    gate3_bias_dtype,
    chunk_g,
    two_tensor_B,
    k_feat,
    n_b,
    n_out,
    n_rowvec,
    n_dual_tiles,
    gate3_n3,
    device_capacity,
):
    """Compile one `DualGatedGemmSm90` configuration against fake tensors. Cached on every argument.

    Every parameter is part of the `@jit_cache` key. **M is the ONLY `cute.sym_int()` symbol; K and
    the N-wide extents are all baked** -- see the body, which opens `m, l = cute.sym_int()` and then
    takes `k, n2` straight from the arguments. This sentence previously claimed K was symbolic; it
    was stale from before the feature extents were baked, and a reader who believed it would
    conclude that `GemmSm90._KEEP_STATIC_LEN_K` is a no-op on this path, which it is not -- the flag
    demonstrably changes this kernel's codegen (SASS `BREAK` 1 -> 0). It is left False here because
    it MEASURES as a regression at D=448 and D=512, not because it does nothing.
    Note that the PRESENCE of each optional term is a key component
    while its value is not -- an absent bias is compiled out entirely rather than added as zero, so
    a kernel built without one cannot be handed one later.

    **Why both feature extents are baked.** A static N folds the epilogue's out-of-bounds bounds to
    literals where a symbolic one predicates per output element -- measured at 2-5% on this family's
    fused sibling -- and a static K does the same for the mainloop's. The artifact cost is what
    makes it affordable: in the TriMul workflow both projections are `(D, D)` against a
    `(tokens, D)` activation, so K and N are the SAME feature dimension and a caller sees one
    artifact per width rather than one per (K, N) pair. The upstream bakes both, commenting each
    `# STATIC`; this is parity, not a new idea.

    The cost lands on a caller who sweeps a feature width, which the workflow does not do at
    runtime -- a model has one.

    Args:
        a_dtype: Cutlass element type of the activation. 16-bit (fp16/bf16).
        b_dtype: Element type of the weights. Must equal `a_dtype`.
        postact_dtype: Element type of the gated output. Must be 16-bit.
        a_major: ``"m"`` or ``"k"`` -- which axis of A is contiguous.
        b_major: ``"n"`` or ``"k"``.
        postact_major: ``"m"`` or ``"n"``. ``"m"`` is the transposed store.
        tile_shape_mn: ``(tile_M, tile_N)``, where tile_N spans the 2N PRE-activation. Must be a
            multiple of 16, and of ``2*chunk_g`` when `chunk_g` > 1.
        cluster_shape_mnk: ``(cluster_M, cluster_N, 1)``.
        pingpong: Whether to use the two-warpgroup ping-pong schedule.
        persistent: Whether the grid is persistent.
        is_dynamic_persistent: Whether tiles are handed out by a GMEM atomic.
        activation: Gate name, a key of `gate_fn_map`. Resolved to the callable here and baked in.
        rowvec_dtype: Element type of the interleaved ``(2N,)`` bias, or None when absent.
        maskvec_dtype: Element type of the ``(M,)`` post-gate mask, or None when absent.
        half_bias_dtype: Element type of the two ``(N,)`` half-width biases, or None when absent.
            Mutually exclusive with `rowvec_dtype` -- both present would double-count the bias.
        gate3_activation: The output gate's activation name, a key of `act_fn_map`, or None when
            there is no output gate. Resolved to the callable here and baked in.
        gate3_bias_dtype: Element type of the ``(N3,)`` output-gate bias, or None when absent.
        chunk_g: The weight layout. Part of the key because it selects the register pairing, the
            epilogue tile width and whether the store permutes.
        two_tensor_B: Whether `Wg` arrives as a separate tensor rather than folded into `B`. Only
            valid with ``chunk_g > 1``.
        n_dual_tiles: Dual work-tile count, or 0 for no output gate. Keyed because the gate decides
            the scheduler's tile count and whether its epilogue branch is traced at all.
        gate3_n3: The output gate's N extent, or 0.
        device_capacity: ``(major, minor)``, re-checked rather than trusted.

    Returns:
        The compiled TVM-FFI entry, callable as
        ``fn(A, B, D, C, epi_args, scheduler_args, [B2])`` with ``D``/``C`` None.

    Raises:
        UnsupportedArchError: If `device_capacity` is not SM90.
        ValueError: Propagated from `GemmSm90` for a tile geometry it refuses.
        AssertionError: Propagated from `GemmGatedMixin` for a `chunk_g` / tile_N combination it
            refuses.
    """
    check_arch_supported(device_capacity)
    m, l = cute.sym_int(), cute.sym_int()
    k, n2 = k_feat, n_b  # M is the only symbolic extent; both feature extents are baked
    div_a, div_b = div_for_dtype(a_dtype), div_for_dtype(b_dtype)
    div_p = div_for_dtype(postact_dtype)
    mA = fake_tensor(a_dtype, (m, k, l), leading_dim=1 if a_major == "k" else 0, divisibility=div_a)
    mB = fake_tensor(
        b_dtype, (n2, k, l), leading_dim=1 if b_major == "k" else 0, divisibility=div_b
    )
    # The post-activation is HALF the pre-activation's N: that is the fold, expressed in the
    # compiled signature so a mis-sized output is a trace-time error rather than a silent overrun.
    mB2 = (
        fake_tensor(b_dtype, (n2, k, l), leading_dim=1 if b_major == "k" else 0, divisibility=div_b)
        if two_tensor_B
        else None
    )

    mPostAct = fake_tensor(
        postact_dtype,
        (m, n_out, l),
        leading_dim=1 if postact_major == "n" else 0,
        divisibility=div_p,
    )
    # The interleaved [bg, bp] bias spans the DUAL pre-activation, which is B's OWN width in exactly
    # one configuration: the element interleave with no output gate. With an output gate B carries
    # the appended W3 rows too, so one symbol would assert `2N == 2N + n3_pad`; with the block
    # interleave B is `Wp`, only N wide, so it would assert `2N == N`. Either way the launch is
    # refused with a shape mismatch naming an argument index rather than the extent. Only the first
    # of the two is reachable from this front door today -- `build_dual_operands` never pairs a row
    # vector with the block interleave -- but the symbol is kept independent because a tied one is
    # a constraint the signature asserts silently.

    mRowVec = fake_tensor(rowvec_dtype, (l, n_rowvec), leading_dim=1, divisibility=4)
    mMask = fake_tensor(maskvec_dtype, (l, m), leading_dim=1, divisibility=4)
    mBiasUp = fake_tensor(half_bias_dtype, (l, n_out), leading_dim=1, divisibility=4)
    mBiasGate = fake_tensor(half_bias_dtype, (l, n_out), leading_dim=1, divisibility=4)
    mPostAct3 = (
        fake_tensor(postact_dtype, (m, gate3_n3, l), leading_dim=1, divisibility=div_p)
        if n_dual_tiles > 0
        else None
    )

    epi_args = DualGatedGemmSm90.EpilogueArguments(
        mPostAct=mPostAct,
        act_fn=gate_fn_map[activation],
        mRowVecBroadcast=mRowVec,
        mMaskColVec=mMask,
        mBiasUp=mBiasUp,
        mBiasGate=mBiasGate,
        mPostAct3=mPostAct3,
        act_fn_3=None if gate3_activation is None else act_fn_map[gate3_activation],
        mRowVecBroadcast3=fake_tensor(
            gate3_bias_dtype, (l, gate3_n3), leading_dim=1, divisibility=4
        ),
        rounding_mode=RoundingMode.RN,
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    return compile_gemm_kernel(
        partial(
            DualGatedGemmSm90,
            chunk_g=chunk_g,
            n_dual_tiles=n_dual_tiles,
            gate3_n3=gate3_n3,
        ),
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        None,  # no D: the gated output IS the post-activation store
        None,  # no C
        epi_args,
        scheduler_args,
        mB2=mB2,
    )


# ────────────── the chunk_g == 1 operand build: weight and bias in ONE launch ──────────────
# Tile of the (N, K) weight one CTA owns. The destination is K-major ``(K, 2N)`` while the sources
# are N-major ``(N, K)``, so a CTA stages its tile through SMEM TRANSPOSED and both the global read
# and the global write stay coalesced. 32x32 gives 1024 threads -- one element each, no inner loop --
# and every access is PREDICATED on both extents, so no N and no K is excluded.
_ILV_BLK_N = 32
_ILV_BLK_K = 32


@cute.kernel
def _interleave_dual_weights_kernel(
    mWg: cute.Tensor,  # (n, k) gate weight
    mWp: cute.Tensor,  # (n, k) up weight
    mB: cute.Tensor,  # (k, 2n) interleaved weight, written
    mBg: cute.Tensor,  # (n_bias,) gate bias, read only when mBias is not None
    mBp: cute.Tensor,  # (n_bias,) up bias, likewise
    mBias: cute.Tensor,  # (1, 2n_pitch) fp32 interleaved bias, written, or None
):
    """Transpose-interleave the two projection weights, and optionally their biases, in one launch.

    Purpose
        The whole host-side cost of the ``chunk_g == 1`` layout, done on the device. Replaces a
        `torch.stack`/`reshape`/`contiguous` chain plus a second chain for the bias plus a widening
        cast per bias vector -- up to four launches -- with one.

    Semantics
        Each CTA owns a ``(_ILV_BLK_N, _ILV_BLK_K)`` tile of the ``(N, K)`` sources and writes
        ``mB[k, 2n] = mWg[n, k]`` and ``mB[k, 2n+1] = mWp[n, k]``. The destination is K-MAJOR while
        the sources are N-major, so the tile is staged through SMEM transposed: the load maps threads
        with K fastest (coalesced read of ``mWg``/``mWp``) and the store maps them with N fastest,
        writing the adjacent gate/up PAIR (a contiguous 2-element store). Without that staging one of
        the two accesses would be strided by ``2N``.

        **Both extents are predicated, so there is no shape condition.** The load and the store each
        test ``n < N`` and ``k < K`` independently, which is what lets an arbitrary ``N`` and ``K``
        take this path rather than falling back -- a tile-multiple requirement would be a shape
        constraint this package does not accept.

        **The operands are rank 2 on purpose.** Under the K-major destination a batched
        ``(l, n, k)`` no longer flattens into the rank-2 case -- the output's leading axis is ``K``,
        which is shared across the batch -- so a rank-3 weight takes the torch reference path
        instead. That is the trade the coalescing is worth: rank 3 has no caller here, while the
        orientation is what every consumer reads.

        With `mBias` present, the ``bk == 0`` CTAs' first thread-row also emits the interleaved pair
        ``mBias[0, 2n] = mBg[n]``, ``mBias[0, 2n+1] = mBp[n]``, WIDENED to fp32 -- which is why the
        caller may pass a 16-bit bias without a separate cast. Row 0's thread additionally zeroes
        the up-to-two pitch-padding elements past ``2*n_bias``, so the destination needs no
        pre-zeroing launch. Rows at or past ``mBg``'s extent emit nothing, which is what lets a
        FLATTENED batched weight share this kernel with an unbatched bias.

    Args:
        mWg: ``(n, k)`` gate weight, k-major. Contiguity in K is assumed, not checked -- a
            k-strided source would still be read as if contiguous, so a wrong descriptor is a wrong
            answer, not a fault.
        mWp: ``(n, k)`` up weight. Must have `mWg`'s shape, dtype and layout; the kernel indexes
            both with one coordinate.
        mB: ``(k, 2n)`` destination, `mWg`'s dtype. Its ``2n`` extent must be exactly twice `mWg`'s
            ``n``, and its ``k`` extent must equal `mWg`'s -- the store predicates on ``mB.shape[0]``
            and on ``mWg.shape[0]``, so a destination short in ``2n`` silently drops columns.
        mBg: ``(n_bias,)`` gate bias, any dtype convertible to fp32, or None. Read only when
            `mBias` is not None; the two are bound at compile time and cannot disagree at launch.
            ``n_bias`` may be smaller than ``n`` (the flattened-batch case) but never larger.
        mBp: ``(n_bias,)`` up bias. Must be present exactly when `mBg` is.
        mBias: ``(1, 2n_pitch)`` fp32 destination with ``2n_pitch >= 2*n_bias``, or None to compile
            the bias out entirely. ``2n_pitch - 2*n_bias`` must be 0 or 2 (a 4-element row pitch
            over an even ``2*n_bias``); a larger pad would leave uninitialized elements inside the
            vectorized load.

    Returns:
        None; `mB` and `mBias` are written in place.
    """
    blk_n = const_expr(_ILV_BLK_N)
    blk_k = const_expr(_ILV_BLK_K)
    tidx, _, _ = cute.arch.thread_idx()
    bn, bk, _ = cute.arch.block_idx()
    n0 = bn * blk_n
    k0 = bk * blk_k

    # Staged TRANSPOSED -- (BLK_K, BLK_N), so the store below reads a whole N-run contiguously.
    smem = cutlass.utils.SmemAllocator()
    sWg = smem.allocate_tensor(
        mWg.element_type, cute.make_ordered_layout((blk_k, blk_n), order=(1, 0)), byte_alignment=16
    )
    sWp = smem.allocate_tensor(
        mWp.element_type, cute.make_ordered_layout((blk_k, blk_n), order=(1, 0)), byte_alignment=16
    )

    # Load: K fastest, so consecutive threads read consecutive K -- contiguous in the (N, K) sources.
    ld_nn = tidx // blk_k
    ld_kk = tidx % blk_k
    n_ld = n0 + ld_nn
    k_ld = k0 + ld_kk
    sWg_t = transpose_view(sWg)  # (BLK_N, BLK_K) view: [nn, kk] IS sWg[kk, nn]
    sWp_t = transpose_view(sWp)
    if n_ld < mWg.shape[0] and k_ld < mWg.shape[1]:
        sWg_t[ld_nn, ld_kk] = mWg[n_ld, k_ld]
        sWp_t[ld_nn, ld_kk] = mWp[n_ld, k_ld]
    cute.arch.barrier()

    # Store: N fastest, writing the adjacent (gate, up) pair into the K-major destination.
    st_kk = tidx // blk_n
    st_nn = tidx % blk_n
    k_st = k0 + st_kk
    n_st = n0 + st_nn
    if k_st < mB.shape[0] and n_st < mWg.shape[0]:
        mB[k_st, 2 * n_st] = sWg[st_kk, st_nn]
        mB[k_st, 2 * n_st + 1] = sWp[st_kk, st_nn]
    if const_expr(mBias is not None):
        # One thread per n (the st_kk == 0 row of the bk == 0 CTAs) emits the widened pair.
        if bk == 0 and st_kk == 0 and n_st < mBg.shape[0]:
            n2 = mBg.shape[0] * 2
            mBias[0, 2 * n_st] = mBg[n_st].to(Float32)
            mBias[0, 2 * n_st + 1] = mBp[n_st].to(Float32)
            if n_st == 0 and n2 < mBias.shape[1]:
                mBias[0, n2] = Float32(0.0)
                mBias[0, n2 + 1] = Float32(0.0)


@cute.jit
def _interleave_dual_weights_jit(
    mWg: cute.Tensor,
    mWp: cute.Tensor,
    mB: cute.Tensor,
    mBg: cute.Tensor,
    mBp: cute.Tensor,
    mBias: cute.Tensor,
    stream,
):
    """Launch `_interleave_dual_weights_kernel` over ``(n, ceil(k/threads), 1)`` CTAs.

    Purpose
        The launch half, separated so the kernel body can be compiled once against symbolic extents
        and re-used for every shape.

    Semantics
        The grid puts ``n`` on x rather than y because y and z are capped at 65535 while x is not,
        and ``n`` -- which carries the flattened batch -- is the extent most likely to be large.

    Args:
        mWg, mWp, mB, mBg, mBp, mBias: As `_interleave_dual_weights_kernel`.
        stream: The CUDA stream. Under ``--enable-tvm-ffi`` this is the environment stream, so the
            compiled callable takes no stream argument.

    Returns:
        None.
    """
    _interleave_dual_weights_kernel(mWg, mWp, mB, mBg, mBp, mBias).launch(
        grid=[
            cute.ceil_div(mWg.shape[0], _ILV_BLK_N),
            cute.ceil_div(mWg.shape[1], _ILV_BLK_K),
            1,
        ],
        block=[const_expr(_ILV_BLK_N * _ILV_BLK_K), 1, 1],
        stream=stream,
    )


@jit_cache
def _compile_dual_weight_interleave(dtype, bg_dtype, bp_dtype):
    """Compile (and cache) the interleave kernel for one ``(weight, bias)`` dtype signature.

    Purpose
        Keeps the interleave off the per-call critical path: every extent is symbolic, so one
        compile serves every shape and the cache hits from the second call onward.

    Semantics
        Divisibility 1 throughout -- deliberately. The kernel is scalar and predicated, so it gains
        nothing from an alignment assumption, and asserting one would make ``K % 8 != 0`` either
        recompile or generate wrong addresses. The result is a single artifact per dtype signature
        that accepts every shape this package accepts.

    Args:
        dtype: The weights' cutlass dtype; also the destination's.
        bg_dtype: The gate bias's cutlass dtype, or None to compile the bias emission out.
        bp_dtype: The up bias's, or None. Must be None exactly when `bg_dtype` is -- the caller
            enforces the pairing; a mismatch here would trace a read of an absent tensor.

    Returns:
        The compiled callable, taking ``(Wg, Wp, B, bg, bp, bias)`` torch tensors directly.
    """
    n_sym, k_sym, n2_sym, nb_sym, pitch_sym = (cute.sym_int() for _ in range(5))
    Wg_cute = fake_tensor(dtype, (n_sym, k_sym))
    Wp_cute = fake_tensor(dtype, (n_sym, k_sym))
    B_cute = fake_tensor(dtype, (k_sym, n2_sym))  # (K, 2N): K-major, NOT (2N, K)
    Bg_cute = fake_tensor(bg_dtype, (nb_sym,))
    Bp_cute = fake_tensor(bp_dtype, (nb_sym,))
    Bias_cute = fake_tensor(Float32, (1, pitch_sym)) if bg_dtype is not None else None
    return cute.compile(
        _interleave_dual_weights_jit,
        Wg_cute,
        Wp_cute,
        B_cute,
        Bg_cute,
        Bp_cute,
        Bias_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _interleave_dual_weights_torch(Wg: Tensor, Wp: Tensor) -> Tensor:
    """Reference column-interleave: transpose both, stack on the LAST axis, then flatten it.

    The ``(K, 2N)`` orientation is the contract -- see `interleave_dual_weights`. Unlike the fused
    path this generalises to a leading batch for free, because the stack-and-reshape is expressed
    on the last two axes either way.
    """
    n, k = Wg.shape[-2], Wg.shape[-1]
    flat = (
        torch.stack([Wg.transpose(-2, -1).contiguous(), Wp.transpose(-2, -1).contiguous()], dim=-1)
        .reshape(*Wg.shape[:-2], k, 2 * n)
        .contiguous()
    )
    # Match the fused path's 8-element row pitch, so both return the same STRIDE as well as the
    # same bytes -- the consumer's `.mT` needs that stride 16-B aligned whichever path produced it.
    pitch = (2 * n + 7) // 8 * 8
    if pitch == 2 * n:
        return flat
    buf = flat.new_empty((*Wg.shape[:-2], k, pitch))
    buf[..., : 2 * n] = flat
    return buf[..., : 2 * n]


def _interleave_dual_bias_torch(bg: Tensor, bp: Tensor) -> Tensor:
    """Reference bias interleave, pitch-aligned to match the fused path. See `interleave_dual_weights`."""
    n2 = 2 * bg.shape[0]
    out = bg.new_zeros((1, (n2 + 3) // 4 * 4), dtype=torch.float32)
    out[0, :n2:2] = bg.float()
    out[0, 1:n2:2] = bp.float()
    return out[:, :n2]


def interleave_dual_weights(
    Wg: Tensor,
    Wp: Tensor,
    bg: Optional[Tensor] = None,
    bp: Optional[Tensor] = None,
    *,
    return_bias: bool = False,
):
    """Build the element-interleaved ``(K, 2N)`` weight -- and bias -- the ``chunk_g == 1`` layout reads.

    Purpose
        In the element-interleave layout, pre-activation column ``2i`` must be gate ``i`` and column
        ``2i+1`` must be up ``i``. This produces that ordering, for the weight and, on request, for
        the projection bias that is added to the same axis.

        **The result is ``(K, 2N)``, so a consumer's ``.mT`` is a free view.** The GEMM wants a
        ``(2N, K)`` N-major B operand; laying the bytes out K-major and transposing the VIEW is what
        makes that cost nothing. Returning ``(2N, K)`` instead would hand the consumer a K-major
        operand -- a different SMEM layout atom, a different cubin -- and a ``.mT`` written against
        this contract would then silently halve the GEMM's N extent instead of being a no-op.

    Semantics
        The whole build is ONE fused kernel launch when the inputs qualify (see below); otherwise a
        torch fallback that produces byte-identical results. That launch is the reason
        ``chunk_g > 1`` is still preferred -- the interleaved form is a function of both weights, so
        a stateless entry point cannot cache it -- but it is now one launch rather than the four a
        16-bit bias used to cost. A caller that owns its weights should still interleave once at
        setup and keep the result.

        Measured against the torch chain it replaces (paired rounds, N x K from 128x128 to
        4096x512, per-call time including the destination allocation):

            weight only     0.97 - 1.03 x   (parity: both are one launch)
            + fp32 bias     0.42 - 0.44 x
            + bf16 bias     0.32 - 0.34 x   (the widening cast is folded in)

        The weight-only column is the one to watch. It is parity rather than a win because there is
        nothing to fuse there -- `torch.stack` is already a single kernel -- so this path must be
        held to NOT REGRESSING. It briefly did: a rank-3 signature cost 4.5 us of `unsqueeze` host
        dispatch against ~1.5 us of device work, i.e. a 1.4x regression on an op whose whole cost is
        dispatch. That is why the kernel takes rank-2 operands and the batched case reshapes.

        The fused path is taken when both weights are contiguous CUDA tensors of the same supported
        dtype, and, for the bias, both bias vectors are contiguous 1-D CUDA tensors of length ``N``.
        Anything else -- CPU, non-contiguous, an unsupported dtype -- falls back to torch. There is
        no shape condition: the kernel predicates its K tail rather than requiring a tile multiple.

    Args:
        Wg: ``(..., N, K)`` gate weight -- rank 2, or rank 3 with a leading batch.
        Wp: ``(..., N, K)`` up weight. Must have the same shape, dtype and device as `Wg`; a
            mismatch is caught here rather than becoming a silently wrong pairing later.
        bg: Optional ``(N,)`` gate bias, in any dtype -- it is widened to fp32 by the kernel, so no
            host cast is needed. Consulted only when `return_bias` is set.
        bp: Optional ``(N,)`` up bias. Must be given exactly when `bg` is: a bias on one projection
            only would bias half the pre-activation, which is not a meaningful operation.
        return_bias: Return the interleaved bias alongside the weight. Keyword-only, and False by
            default so the plain two-argument call still returns a single tensor.

    Returns:
        Without `return_bias`, a new contiguous ``(..., K, 2N)`` tensor with gate COLUMNS at even
        indices and up columns at odd ones. A batched weight keeps its leading modes, but only via
        the torch path -- the fused kernel is rank 2.

        With `return_bias`, the pair ``(B, bias)``, where `bias` is None if `bg` is, and otherwise a
        ``(1, 2N)`` fp32 **view of a row-pitch-padded buffer**, ready for the epilogue's 4-element
        vectorized broadcast load. Returning it already aligned is deliberate: passing it through
        `_pitch_align_broadcast` would cost a second launch whenever ``N`` is odd. The caller must
        keep the returned view alive -- it owns the only reference to the padded buffer.

    Raises:
        AssertionError: If the two weights disagree in shape, dtype or device, or if exactly one of
            `bg` / `bp` is given, or either is not ``(N,)``.
    """
    assert Wg.shape == Wp.shape, f"gate/up weights must agree in shape: {Wg.shape} vs {Wp.shape}"
    assert Wg.dtype == Wp.dtype and Wg.device == Wp.device
    assert (bg is None) == (bp is None), "bg and bp must both be given or both omitted"
    n, k = Wg.shape[-2], Wg.shape[-1]
    want_bias = return_bias and bg is not None
    if want_bias:
        assert bg.shape == (n,) and bp.shape == (n,), (
            f"bg and bp must both be (N={n},); got {tuple(bg.shape)} and {tuple(bp.shape)}"
        )
    fusable = (
        Wg.is_cuda
        and Wg.is_contiguous()
        and Wp.is_contiguous()
        and Wg.dtype in torch2cute_dtype_map
        and Wg.dim() == 2  # rank 3 has no flat form under a K-major destination -- torch path
        and (
            not want_bias
            or (
                bg.is_cuda
                and bp.is_cuda
                and bg.is_contiguous()
                and bp.is_contiguous()
                and bg.dtype in torch2cute_dtype_map
                and bp.dtype in torch2cute_dtype_map
            )
        )
    )
    if not fusable:
        B = _interleave_dual_weights_torch(Wg, Wp)
        return (B, _interleave_dual_bias_torch(bg, bp) if want_bias else None) if return_bias else B

    n2 = 2 * n
    # PITCH-PADDED to 8 elements (16 B at 16-bit). The consumer transposes this to the (2N, K)
    # n-major operand, whose leading stride is then this pitch -- and a TMA descriptor requires that
    # stride 16-B aligned. Padding the PITCH rather than the extent is what keeps an arbitrary N
    # (137, 97, ...) legal; without it the orientation would impose N % 4 == 0. Same trick, same
    # reason, as `_pitch_align_broadcast`. The pad columns are never written and never read: the
    # returned view stops at 2N, so no descriptor addresses them.
    B_buf = torch.empty((k, (n2 + 7) // 8 * 8), dtype=Wg.dtype, device=Wg.device)
    # One over-allocated buffer, its tail zeroed BY THE KERNEL, so the pitch alignment the epilogue
    # load needs never costs a second launch. See `_pitch_align_broadcast` for why the pitch and not
    # the extent is what gets padded.
    bias_buf = (
        torch.empty((1, (n2 + 3) // 4 * 4), dtype=torch.float32, device=Wg.device)
        if want_bias
        else None
    )
    _compile_dual_weight_interleave(
        torch2cute_dtype_map[Wg.dtype],
        torch2cute_dtype_map[bg.dtype] if want_bias else None,
        torch2cute_dtype_map[bp.dtype] if want_bias else None,
    )(Wg, Wp, B_buf, bg if want_bias else None, bp if want_bias else None, bias_buf)
    B = B_buf[:, :n2]
    if not return_bias:
        return B
    return B, (bias_buf[:, :n2] if want_bias else None)


def append_gate3_weight(B: Tensor, W3: Tensor, tile_N: int) -> Tensor:
    """Append the output gate's weight rows to the dual weight, padded to a whole work tile.

    Purpose
        The one host-side step the fused output gate needs, shared by every front door that offers
        it. Making the gate a REGION of the same operand is what lets an output-gate work tile be an
        ORDINARY work tile: same accumulator, same shared memory, same scheduler stream.

    Semantics
        Returns a NEW ``(2N + n3_pad, K)`` tensor with ``B``'s rows first and ``W3``'s next, the tail
        zero-filled to a multiple of `tile_N`. The padding rows are never stored -- the gate's TMA
        descriptor carries the true ``N3`` extent and predicates them -- but they must be ZERO
        rather than uninitialized: their accumulator is contracted like any other, so garbage there
        would propagate into a real tile's neighbours through the shared staging buffer.

        This is an allocation and two copies **per call**, which is the gate's dominant cost at
        small shapes (measured: 3.9x the ungated kernel for 1.5x the work, essentially all of it
        here). A caller invoking the gate in a loop should hoist it and pass the combined weight to
        a plain ``chunk_g=1`` launch instead.

    Args:
        B: The ``(2N, K)`` dual weight -- already interleaved for ``chunk_g == 1``. Only that layout
            is supported: with ``chunk_g > 1`` the two weights are separate TMA tensors and there is
            nothing to append to.
        W3: The ``(N3, K)`` output-gate weight. Must share `B`'s dtype, device and ``K``; a mismatch
            is a silently wrong gate, since the kernel sees one matrix either way.
        tile_N: The CTA tile over the 2N pre-activation. ``2N`` must already be a multiple of it --
            the region boundary has to fall on a work-tile edge, or one tile would straddle the two
            regions and take a single epilogue arm for columns belonging to both.

    Returns:
        The combined ``(2N + n3_pad, K)`` weight.
    """
    n2, K = B.shape[-2], B.shape[-1]
    assert n2 % tile_N == 0, f"2N={n2} must be a multiple of tile_N={tile_N}"
    assert W3.shape[-1] == K, f"W3 must be (n3, k={K}); got {tuple(W3.shape)}"
    n3 = W3.shape[-2]
    n3_pad = (n3 + tile_N - 1) // tile_N * tile_N
    B_all = B.new_zeros((n2 + n3_pad, K))
    B_all[:n2] = B
    B_all[n2 : n2 + n3] = W3
    return B_all


def _pitch_align_broadcast(t: Optional[Tensor]) -> Optional[Tensor]:
    """Give an epilogue broadcast vector a 4-element-aligned row pitch, copying only if it lacks one.

    Purpose
        Removes a shape constraint rather than inheriting one. The epilogue's broadcast load is
        4-element vectorized, so a ``(l, x)`` vector's row pitch -- ``x`` -- must be a multiple of
        4 (see `check_broadcast_alignment`). Taken literally that would make a per-row mask impose
        ``M % 4 == 0``, and a per-column bias impose ``N % 4 == 0``, on extents this package
        otherwise leaves free: its only permitted shape constraint is the 16-byte input alignment.
        Padding the PITCH here keeps the extent arbitrary.

    Semantics
        A no-op returning the input unchanged when the pitch already divides 4 -- which is the
        common case, so the aligned path stays zero-copy and byte-identical. Otherwise allocates a
        ``(l, ceil(x/4)*4)`` buffer, copies the data into its leading columns, and returns a
        ``(l, x)`` VIEW of it whose row stride is the padded pitch. The padding columns are never
        read: the load is predicated to the true extent.

        The cost is one O(x) allocation and copy, against a GEMM that is O(M*N*K) -- and it is
        bounded by the vector's own length, so it cannot violate the package's rule that scratch
        along a token axis stays O(N_token).

    Args:
        t: A rank-2 ``(l, x)`` broadcast vector, or None (returned unchanged).

    Returns:
        `t` itself when already aligned, else a `(l, x)` view of a pitch-padded copy. The returned
        tensor must be kept alive by the caller for the duration of the launch -- it owns the only
        reference to the padded buffer.
    """
    if t is None or t.shape[-1] % 4 == 0:
        return t
    l, x = t.shape
    padded = t.new_zeros((l, (x + 3) // 4 * 4))
    padded[:, :x] = t
    return padded[:, :x]


class DualOperands(NamedTuple):
    """The B-side operands and bias terms a dual-gated GEMM launch takes.

    Attributes:
        B: The first weight operand -- the interleaved ``(2N, K)`` weight for ``chunk_g == 1``
            (with the output gate's rows already appended when there is one), or ``Wp`` itself for
            the block-interleave layout.
        B2: The second weight operand, ``Wg``, for the block-interleave layout; None otherwise.
        rowvec_bias: The ``(1, 2N)`` fp32 element-interleaved ``[bg, bp]`` bias for
            ``chunk_g == 1``, or None. Never set at the same time as `half_up` / `half_gate` --
            both arriving would double-count the bias, since each is added to the whole
            pre-activation.
        half_up: The ``(1, N)`` fp32 up-projection bias for the block-interleave layout's direct
            two-vector load, or None.
        half_gate: The ``(1, N)`` fp32 gate-projection bias, likewise. Present exactly when
            `half_up` is.
    """

    B: Tensor
    B2: Optional[Tensor]
    rowvec_bias: Optional[Tensor]
    half_up: Optional[Tensor]
    half_gate: Optional[Tensor]


def build_dual_operands(
    Wg: Tensor,
    Wp: Tensor,
    bg: Optional[Tensor],
    bp: Optional[Tensor],
    *,
    chunk_g: int,
    tile_N: int,
    W3: Optional[Tensor] = None,
) -> DualOperands:
    """Build the weight operand(s) and the bias term(s) for a dual-gated GEMM launch.

    Purpose
        The one place either front door -- the plain dual-gated GEMM or its LayerNorm-fused sibling
        -- turns ``(Wg, Wp, bg, bp)`` into what the kernel is launched with. Shared because it is
        layout work, not LayerNorm work, and because the bias path below is a MEASURED choice that
        must not be allowed to drift between two copies of it.

    Semantics
        Two weight layouts, selected by `chunk_g`:

        * ``chunk_g == 1`` (element interleave) -- ``Wg`` and ``Wp`` are row-interleaved into a
          single ``(2N, K)`` operand, and the bias, if any, is interleaved to match IN THE SAME
          LAUNCH (`interleave_dual_weights`). An output gate's rows are then appended.
        * ``chunk_g > 1`` (block interleave) -- the weights are passed straight through and the
          kernel interleaves ``chunk_g``-wide blocks in shared memory, so there is no host build at
          all.

        In the BLOCK-interleave layout the bias has two possible paths, and this takes the direct
        one UNCONDITIONALLY -- the two ``(N,)`` vectors are loaded as they are and scattered into
        their register halves by `ChunkedHalfBiasLoad`, with no host build at all. The alternative,
        one interleaved ``(2N,)`` vector read by the single vectorized row-vector load, is NOT
        implemented, and that is a measured decision rather than an omission.

        **Why there is no size-dependent switch here.** The kernel this reproduces has one, at
        ``M*N = 2**25``, on the grounds that the two-vector epilogue costs ~13% more KERNEL time and
        that at large ``M*N`` this outweighs a host build that is off the critical path. Its
        crossover was measured on an H200. Re-measured on an H100 (paired, interleaved rounds via
        `bench_timing.benchmark_paired`, plus profiler kernel-only time), the effect has the
        OPPOSITE SIGN: the direct path is faster at every size tried, by a flat ~7% of kernel time
        with no trend across a 1024x span of ``M*N``.

            N     M*N        kernel direct/rowvec     end-to-end direct/rowvec
            128   2.6e5 .. 1.3e8    0.930 .. 0.951        0.735 .. 0.907
            256   5.2e5 .. 2.7e8    0.928 .. 0.941        0.694 .. 0.921
            384   3.1e6 .. 1.0e8    0.918 .. 0.935        0.728 .. 0.899
            512   4.2e6 .. 1.3e8    0.930 .. 0.936        0.698 .. 0.908
            1024  8.4e6 .. 2.7e8    0.918 .. 0.935        0.693 .. 0.917

        So porting the constant would have made every large-``M*N`` call ~7% slower on kernel time
        and ~8% slower end-to-end. Both arms were verified BITWISE equal first, so this is a pure
        performance comparison. If a future arch flips the sign, re-measure it there -- do not
        inherit either number.

        The bias is widened to fp32 REGARDLESS of what the caller passed, and that is a PERFORMANCE
        decision, not a numerical one: the epilogue converts to fp32 before adding either way, so a
        16-bit vector produces bit-identical output -- it is just slower to load. Measured on an
        H100 at 4096x256x128, block interleave: a bf16 pair through `ChunkedHalfBiasLoad` costs
        6.53 us against 5.30 for the fp32 pair -- **1.23x**, on a kernel whose no-bias form is 4.84.
        A 4-element O(N) cast against an O(M*N*K) GEMM is not a cost worth exposing that cliff for.

    Args:
        Wg: ``(N, K)`` gate weight, 16-bit. Must be contiguous to take the fused interleave; a
            non-contiguous weight still produces the right answer, through a slower torch path.
        Wp: ``(N, K)`` up weight, same shape/dtype/device as `Wg`.
        bg: Optional ``(N,)`` gate bias, any dtype.
        bp: Optional ``(N,)`` up bias. Must be given exactly when `bg` is -- checked by the callers,
            whose error message names the front-door argument; here it is an assertion.
        chunk_g: The weight layout. 1 or a multiple of 16; the callers validate it.
        tile_N: The CTA tile over the 2N pre-activation, needed only to pad an output gate's weight
            to a work-tile edge. Ignored without `W3`.
        W3: Optional ``(N3, K)`` output-gate weight, appended to the interleaved dual weight. Only
            valid with ``chunk_g == 1``; with the block interleave there is no single operand to
            append to, and the callers refuse that combination before reaching here.

    Returns:
        A `DualOperands` bundle. Every tensor in it must be kept alive by the caller until the
        launch completes: the interleaved weight and the padded bias buffers are freshly allocated
        here and are referenced only by the returned views.
    """
    assert (bg is None) == (bp is None), "bg and bp must both be given or both omitted"
    n = Wg.shape[-2]
    rowvec_bias = half_up = half_gate = None
    if chunk_g > 1:
        # Block interleave: the weights are loaded directly, Wp as B and Wg as the second tensor.
        B, B2 = Wp, Wg
        if bg is not None:
            half_up = _pitch_align_broadcast(bp.float().reshape(1, -1).contiguous())
            half_gate = _pitch_align_broadcast(bg.float().reshape(1, -1).contiguous())
    else:
        B, rowvec_bias = interleave_dual_weights(Wg, Wp, bg, bp, return_bias=True)
        # The helper lays the bytes out K-major ``(K, 2N)``; the GEMM wants the N-major ``(2N, K)``
        # operand, and `.mT` is that view for free. Placing it HERE -- immediately at the call, as
        # the kernel this ports from does -- is what keeps every downstream `B.shape[-2]` reading
        # ``2N``: `append_gate3_weight` below, and the two baked compile-key extents in the callers.
        B = B.mT
        B2 = None
        if W3 is not None:
            # Region-aware output gate: append W3's rows so a gate tile is an ORDINARY work tile of
            # the same operand. The append is layout work, not LayerNorm work, and one copy is what
            # keeps the two front doors from drifting.
            B = append_gate3_weight(B, W3, tile_N)
    return DualOperands(B, B2, rowvec_bias, half_up, half_gate)


def _as_batched(t: Optional[Tensor], name: str) -> Optional[Tensor]:
    """Give a 2-D operand the leading batch mode the kernel addresses it with.

    Purpose
        The kernel's operands are always rank 3 ``(l, ...)``; a caller with a single batch holds
        rank 2. Converting once, here, keeps every downstream helper (`perm3d_single`, `get_major`)
        on one rank instead of branching.

    Args:
        t: The operand, rank 2 or rank 3, or None (returned unchanged so optional operands need no
            guard at the call site).
        name: The operand's name, used only in the error message.

    Returns:
        A rank-3 **view** -- `unsqueeze` does not copy, so writes through the result reach the
        caller's tensor, which is what makes the in-place output work.

    Raises:
        ValueError: If `t` is neither rank 2 nor rank 3. Named rather than left to fail later as a
            shape mismatch inside the compiled entry, which reports symbol names rather than
            arguments.
    """
    if t is None:
        return None
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim == 3:
        return t
    raise ValueError(f"{name} must be 2-D (m, k) or 3-D (l, m, k); got shape {tuple(t.shape)}.")


# ───────────────────────────── autotuning: the declared space ─────────────────────────────
# Reproduced from `main`'s `gated_gemm_gate._ggg_perf_configs` / `_ggg_prune`, verbatim including
# what it does NOT sweep. `main`'s comment records why, and the measurement behind it is the part
# worth carrying: **tile_M is pinned at 128** because tile_M=256 measured 6-14x slower and 192 is
# invalid for this epilogue, and tile_N is left to the caller. So `chunk_g` is the SOLE knob, and a
# pool that added tiles here would be sweeping configurations `main` already measured and rejected.


def dual_gated_tuning_space() -> AxisSpace:
    """The declared autotuning axis for `dual_gated_gemm` -- exactly one, reproducing `main`.

    Purpose
        Makes "the only tuned knob is the weight layout" readable off the kernel rather than
        inferable from a two-element list.

    Semantics
        One axis, `chunk_g`, over {1, 16}, named for the parameter it sets. 1 is the element
        interleave, which needs a host-side interleave launch on every call; 16 is the block
        interleave, which loads `Wg`/`Wp` directly through a two-tensor TMA and needs no host
        preparation. Which wins is shape-dependent, which is why it is measured rather than
        defaulted.

    Returns:
        A fresh `AxisSpace`, so a caller may inspect or subset it without touching the decorated
        pool.
    """
    return AxisSpace(
        TuneAxis(
            "chunk_g",
            domain="1 (element interleave) or a multiple of 16 (block interleave); "
            "GemmGatedMixin.maybe_override_epi_tile asserts the rest",
            values=(1, 16),
        )
    )


#: The pool the tuned path sweeps. Two points -- see the note above for why it is not larger.
DUAL_GATED_TUNING_SPACE = dual_gated_tuning_space()


def dual_gated_config_is_valid(config, request) -> bool:
    """Whether one candidate CAN RUN for this request. `main`'s `_ggg_prune`, ported unchanged.

    Purpose
        The block-interleave layout is not universally applicable, and the failure if it is applied
        anyway is not a slow kernel -- it is a refused compile or a mis-paired register fold.

    Semantics
        Pure function of the config and the request, as `ConfigSpace` requires: the surviving pool
        must be identical on every rank, or ranks measure different candidate sets and the consensus
        step compares numbers for different kernels.

        `chunk_g == 16` needs both the contraction dim K and the projection width N to be multiples
        of 16. `chunk_g == 1` runs any shape the front door accepts, so the pool can never empty --
        which matters, because an empty pool leaves the kernel with nothing to run.

    Args:
        config: The candidate; `config["chunk_g"]` is the layout.
        request: The bound call arguments by name. Reads `A` (for K) and `Wg` (for N).

    Returns:
        True if this layout can run for these operands.
    """
    if config["chunk_g"] == 1:
        return True
    K = request["A"].shape[-1]
    N = request["Wg"].shape[-2]
    return K % 16 == 0 and N % 16 == 0


def heuristic_chunk_g(K: int, N: int, device=None) -> int:
    """Pick `dual_gated_gemm`'s weight layout without timing: 16 (block) or 1 (element).

    Purpose
        Give a caller that has no measured preference the layout `main` uses. `chunk_g > 1` is not
        a speed knob alone — it decides whether the call LAUNCHES A KERNEL at all: at ``chunk_g=1``
        the entry must build the element-interleaved ``(2N, K)`` weight per call, while
        ``chunk_g=16`` hands `Wg` and `Wp` to a two-tensor TMA and builds nothing. Leaving this at
        the ``chunk_g=1`` default is what made this package launch a per-call interleave kernel
        that `main` does not launch at all.

    Functionality & semantics
        16 whenever both extents admit it, else 1. The 16 is the stmatrix atom's N-width, so a
        block narrower than that has no register pairing to exploit; the two extents must each be a
        multiple of it for the block-interleaved pairing to tile evenly.

        **The choice is output-invariant.** `chunk_g` changes the layout of the PRE-activation
        along the internal ``2N`` axis and the register pairing that consumes it; the ``(m, n)``
        post-activation this entry returns is the same values in the same order either way. That is
        what makes it selectable by heuristic rather than by the caller's data layout — and it is
        why a wrong answer here is a perf bug, never a correctness one.

        ARCH-AWARE, perf layer only: the prefer-16 rule was measured on
        `heuristic_arch.TUNED_ARCH`. On any other arch this warns ONCE and still returns 16 — the
        warning says the THRESHOLD may be untuned, not that the value is invalid. The
        16-divisibility test itself is arch-independent.

    Args:
        K: The contraction extent (`A`'s last dim). Must be a positive int.
        N: The GLU output width, i.e. `Wg.shape[0]`. NOT the interleaved ``2N``. Passing ``2N`` here
            only ever loosens the test (an even multiple of 16 stays one), so a mix-up does not
            raise -- it silently picks 16 on a shape that cannot tile it, and the kernel then
            raises from `GemmGatedMixin`. Pass the same value you pass as the weight's leading dim.
        device: Selects the arch for the warning only. ``None`` suppresses the warning and is
            byte-identical in the returned value.

    Returns:
        ``16`` when ``K % 16 == 0 and N % 16 == 0``, else ``1``. The caller must still satisfy the
        kernel's own `chunk_g > 1` requirements -- ``tile_N % (2 * chunk_g) == 0``,
        ``cluster_N == 1``, and no fused ``W3`` -- which this function does not see and cannot check.
    """
    if device is not None:
        arch = heuristic_arch(device)
        if arch != TUNED_ARCH:
            warn_arch_suboptimal_once(arch, "dual_gated_gemm")
    return 16 if K % 16 == 0 and N % 16 == 0 else 1


@autotune(
    space=DUAL_GATED_TUNING_SPACE,
    key=["activation", "pingpong", "persistent", "is_dynamic_persistent"],
    validity=dual_gated_config_is_valid,
    gate="do_autotune",
)
def dual_gated_gemm(
    A: Tensor,  # (m, k) or (l, m, k)
    Wg: Tensor,  # (n, k)
    Wp: Tensor,  # (n, k)
    PostAct: Tensor,  # (m, n), or (n, m) when transposed
    tile_M: int,
    tile_N: int,
    cluster_M: int = 1,
    cluster_N: int = 1,
    *,
    bg: Optional[Tensor] = None,  # (n,)
    bp: Optional[Tensor] = None,  # (n,)
    mask: Optional[Tensor] = None,  # (m,)
    W3: Optional[Tensor] = None,  # (n3, k)
    b3: Optional[Tensor] = None,  # (n3,)
    PostAct3: Optional[Tensor] = None,  # (m, n3)
    chunk_g: int = 1,
    activation: str = "glu",
    gate3_activation: str = "sigmoid",
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    tile_count_semaphore: Optional[Tensor] = None,  # (1,)
    max_swizzle_size: int = 8,
    do_autotune: bool = False,
) -> None:
    """Dual-gated GEMM ``out = gate_fn(A @ Wg^T + bg, A @ Wp^T + bp) * mask``, written into PostAct.

    Note the transposes: both weights are stored ``(n, k)``, so the contraction is over the LAST
    axis of every operand. That is the layout WGMMA wants; pass the weights in that shape rather
    than transposing a ``(k, n)`` tensor, whose non-unit stride the TMA descriptor cannot use.

    Args:
        A: ``(m, k)`` activation, **already normalized** -- this entry fuses no LayerNorm. Must be
            on a CUDA device (the one whose capability is gated), 16-bit (fp16/bf16), contiguous,
            and its ``k`` extent must satisfy the package's 16-byte alignment floor. Not modified.
        Wg: ``(n, k)`` gate-projection weight, same dtype and device as `A`.
        Wp: ``(n, k)`` up-projection weight, same shape, dtype and device as `Wg`.
        PostAct: The output, **written in place** so the caller owns the allocation. Its LOGICAL
            shape is always ``(m, n)``; the transposed store is selected by its STRIDES, not by a
            different shape. So an n-major output is ``torch.empty(m, n)``, and an m-major one is
            ``torch.empty(n, m).T`` -- a transposed view whose logical shape is still ``(m, n)``.
            Passing a raw ``(n, m)`` tensor is a shape mismatch reported against a symbol name, not
            a transposed store. Must be 16-bit and preallocated at exactly this shape; nothing
            resizes it, and a wrong ``n`` writes past the fold's half-width tile.
        tile_M: CTA tile M. One of the geometries `GemmSm90.__init__` accepts.
        tile_N: CTA tile N over the **2N pre-activation**, so the output tile is ``tile_N // 2``.
            Must be a multiple of 16, and of ``2*chunk_g`` when ``chunk_g > 1``.
        cluster_M: Threadblock-cluster extent along M. Power of two; ``cluster_M * cluster_N <= 8``.
        cluster_N: Cluster extent along N. Must be 1 when ``chunk_g > 1`` -- the two-tensor load
            does not multicast B.
        bg: Optional ``(n,)`` gate bias, added to the pre-activation before the gate.
        bp: Optional ``(n,)`` up bias. Must be present exactly when `bg` is; one without the other
            would bias only half the pre-activation.
        mask: Optional ``(m,)`` per-row multiplier applied to the fp32 output AFTER the gate. It
            cannot be folded into `bg`/`bp`: the gate is nonlinear, so masking the input is a
            different function.
        W3: Optional ``(n3, k)`` output-gate weight. Its rows are APPENDED to the dual weight, so
            ``out3 = gate3_fn(A @ W3^T + b3)`` costs no second mainloop -- an output-gate work tile
            is an ORDINARY work tile of a wider operand. Requires ``chunk_g == 1`` and
            ``2*n % tile_N == 0``: the region boundary must fall on a work-tile edge, or one tile
            would straddle both regions and take a single epilogue arm for columns belonging to
            each. See `append_gate3_weight` for the per-call host cost.
        b3: Optional ``(n3,)`` output-gate bias, added before the gate's activation.
        PostAct3: The output gate's result, ``(m, n3)`` row-major, **written in place**. Required
            exactly when `W3` is given -- the gate is a compile-time feature, so it cannot be
            requested without a destination.
        chunk_g: Weight layout. 1 interleaves the two weights on the host every call; a multiple of
            16 loads them directly and is preferred wherever ``n`` and ``k`` allow it.
        activation: Gate name; a key of `gate_fn_map`.
        gate3_activation: The output gate's activation; a key of `act_fn_map`. Ignored without
            `W3`.
        pingpong: Use the two-warpgroup ping-pong schedule. Requires `persistent`.
        persistent: Launch one resident wave that loops over work tiles.
        is_dynamic_persistent: Hand out tiles through a GMEM atomic instead of statically.
        tile_count_semaphore: A **zeroed** ``int32`` tensor of shape ``(1,)``. Required when
            `is_dynamic_persistent`. A stale value makes the grid skip tiles, silently truncating
            the output.
        max_swizzle_size: Scheduler rasterization swizzle width. A performance knob only.
        do_autotune: Choose `chunk_g` by MEASUREMENT instead of taking it from the caller.
            Keyword-only, like every argument after ``cluster_N``. Left False (the default) this is
            exactly the fixed-config entry it has always been, and the gate costs one dict lookup,
            so a pinned perf cell still times a kernel rather than a sweep. Set True and `chunk_g`
            must NOT be passed: pinning the one knob the tuner sweeps would make the config that was
            measured and the config that ran differ. The winner is readable afterwards as
            ``dual_gated_gemm.autotuner.best_config``.
        do_autotune: Choose `chunk_g` by MEASUREMENT instead of taking it from the caller.
            Keyword-only. Left False (the default) this is exactly the fixed-config entry it has
            always been, and the gate costs one dict lookup, so a pinned perf cell still times a
            kernel rather than a sweep. Set True and `chunk_g` must NOT be passed: pinning the one
            knob the tuner sweeps would make the config that was measured and the config that ran
            differ. The winner is readable afterwards as ``dual_gated_gemm.autotuner.best_config``.

    Returns:
        None. The result is in `PostAct`.

    Raises:
        UnsupportedArchError: If the device is not SM90.
        NotImplementedError: For ``W3`` with ``chunk_g > 1`` -- that combination needs a THIRD B
            operand in the kernel signature, which this package's ``GemmSm90.__call__`` does not
            carry.
        ValueError: For any front-door violation -- an unsupported dtype, a shape disagreement, a
            bias supplied on one projection only, an output-gate request whose region boundary is
            not tile-aligned or whose destination is missing, `is_dynamic_persistent` without a
            semaphore, a `chunk_g` that is neither 1 nor a multiple of 16, or an unknown
            `activation`. Also propagated from `GemmSm90` for a tile geometry it refuses.
        AssertionError: Propagated from `GemmGatedMixin` for a `chunk_g` / `tile_N` combination it
            refuses.
    """
    device_capacity = require_sm90(A.device)
    if activation not in gate_fn_map:
        as_gate_fn(activation)  # raises with the available names
    # Check order is part of the contract: the declared unsupported regions in
    # tests/kernels/test_dual_gated_gemm.py are matched first-wins, so a combo that violates two
    # rules must report the one listed first there. Keep the two orders in step -- dtype, then
    # chunk_g, then the bias pairing, then the cluster width.
    #
    # EVERY tensor argument goes through `check_tensor`, not only the ones the kernel feeds to a
    # TMA. Which of them is an "operand" and which an epilogue broadcast term is a fact about the
    # kernel's plumbing that this signature does not expose, so it must not decide whether a caller
    # mistake is reported or left to fail as a `KeyError` from a compile-key lookup.
    m, k = A.shape[-2], A.shape[-1]
    n = Wg.shape[-2]
    # DTYPES first, EXTENTS after the configuration checks below. The split is the check ORDER the
    # declared unsupported regions mirror: a dtype the kernel has no atom for is a deeper refusal
    # than a knob set wrong, while a mis-sized argument is shallower than one. Keep the two in step.
    for _nm, _t in (
        ("A", A),
        ("Wg", Wg),
        ("Wp", Wp),
        ("PostAct", PostAct),
        ("W3", W3),
        ("PostAct3", PostAct3),
    ):
        check_tensor(_nm, _t, expect_width=2)
    if chunk_g != 1 and chunk_g % 16 != 0:
        raise ValueError(
            f"chunk_g must be 1 (element interleave) or a multiple of 16 (block interleave); got "
            f"{chunk_g}. 16 is the stmatrix atom N-width."
        )
    if (bg is None) != (bp is None):
        raise ValueError(
            "bg and bp must both be given or both omitted: a bias on one projection only would "
            "bias half the pre-activation, which is not a meaningful operation."
        )
    if is_dynamic_persistent and tile_count_semaphore is None:
        raise ValueError(
            "is_dynamic_persistent=True requires tile_count_semaphore: the SM90 dynamic scheduler "
            "hands out work tiles through an atomic counter in GMEM. Pass a zeroed int32 (1,)."
        )
    if chunk_g > 1 and cluster_N != 1:
        raise ValueError(
            f"chunk_g>1 loads Wg and Wp through a two-tensor TMA that does not multicast B, so it "
            f"requires cluster_N=1; got cluster_N={cluster_N}."
        )
    if (W3 is None) != (PostAct3 is None):
        raise ValueError(
            f"W3 and PostAct3 describe one output and must be given together; got "
            f"W3={'a tensor' if W3 is not None else None}, "
            f"PostAct3={'a tensor' if PostAct3 is not None else None}."
        )

    for _nm, _t in (("bg", bg), ("bp", bp), ("b3", b3), ("mask", mask)):
        check_tensor(_nm, _t)
    two_tensor_B = chunk_g > 1
    n2 = 2 * Wg.shape[-2]
    n_dual_tiles, gate3_n3 = 0, 0
    if W3 is not None:
        if two_tensor_B:
            raise NotImplementedError(
                "the fused output gate (W3) currently requires chunk_g=1. With chunk_g>1 the dual "
                "weights arrive as two separate TMA tensors, so W3 cannot be appended to them and "
                "needs a THIRD B operand -- a slot GemmSm90.__call__ does not carry. Use "
                "chunk_g=1, or run the output gate as a separate launch."
            )
        if n2 % tile_N != 0:
            raise ValueError(
                f"the fused output gate appends W3's rows after the dual weight, so the region "
                f"boundary must fall on a work-tile edge: 2*n (={n2}) must be a multiple of tile_N "
                f"(={tile_N}); got remainder {n2 % tile_N}."
            )
        n_dual_tiles, gate3_n3 = n2 // tile_N, W3.shape[-2]

    mask_p = _pitch_align_broadcast(None if mask is None else mask.reshape(1, -1))
    check_broadcast_alignment("mask", mask_p)

    # EXTENTS, after the configuration checks: a mis-sized argument is the shallowest refusal here,
    # so it is reported only once the kernel the caller asked for is known to exist.
    check_tensor("A", A, expect_shape=(m, k))
    check_tensor("Wg", Wg, expect_shape=(n, k))
    check_tensor("Wp", Wp, expect_shape=(n, k))
    check_tensor("PostAct", PostAct, expect_shape=(m, n))
    check_tensor("W3", W3, expect_shape=(None, k))
    if W3 is not None:
        check_tensor("PostAct3", PostAct3, expect_shape=(m, W3.shape[-2]))
        check_tensor("b3", b3, expect_shape=(W3.shape[-2],))
    check_tensor("bg", bg, expect_shape=(n,))
    check_tensor("bp", bp, expect_shape=(n,))
    check_tensor("mask", mask, expect_shape=(m,))

    # The interleaved (2N,) bias and the two half-width (N,) vectors are mutually exclusive: the
    # element-interleave layout has no register half-split to scatter into, and the block-interleave
    # layout would double-count if both arrived. `build_dual_operands` owns that choice, and the
    # fp32 widening and the M*N bias-path crossover with it.
    B, B2, rowvec_bias, half_up, half_gate = build_dual_operands(
        Wg, Wp, bg, bp, chunk_g=chunk_g, tile_N=tile_N, W3=W3
    )

    A_p = perm3d_single(_as_batched(A, "A"))
    B_p = perm3d_single(_as_batched(B, "Wp" if two_tensor_B else "the interleaved weight"))
    PostAct_p = perm3d_single(_as_batched(PostAct, "PostAct"))
    B2_p = perm3d_single(_as_batched(B2, "Wg")) if two_tensor_B else None
    PostAct3_p = perm3d_single(_as_batched(PostAct3, "PostAct3")) if PostAct3 is not None else None
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    postact_major = get_major(PostAct_p, "m", "n")

    # fp32 for the same reason as the projection biases (see `build_dual_operands`), and it is the
    # dtype the kernel this reproduces REQUIRES outright (`assert b3.dtype == torch.float32`).
    # Accepting a 16-bit b3 and passing it through would put `Gate3RowVecLoad` on the same narrow
    # load the half-bias pair was measured 1.23x slower on.
    b3_p = _pitch_align_broadcast(None if b3 is None else b3.float().reshape(1, -1).contiguous())
    check_broadcast_alignment("rowvec_bias", rowvec_bias)
    check_broadcast_alignment("bg", half_gate)
    check_broadcast_alignment("bp", half_up)
    check_broadcast_alignment("b3", b3_p)

    compiled_fn = _compile_dual_gated_gemm(
        torch2cute_dtype_map[A.dtype],
        torch2cute_dtype_map[B.dtype],
        torch2cute_dtype_map[PostAct.dtype],
        a_major,
        b_major,
        postact_major,
        (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        activation,
        None if rowvec_bias is None else torch2cute_dtype_map[rowvec_bias.dtype],
        None if mask is None else torch2cute_dtype_map[mask.dtype],
        None if half_up is None else torch2cute_dtype_map[half_up.dtype],
        None if W3 is None else gate3_activation,
        None if b3_p is None else torch2cute_dtype_map[b3_p.dtype],
        chunk_g,
        two_tensor_B,
        A.shape[-1],
        # The three N-wide extents, BAKED. Read off the tensors that will actually be passed rather
        # than re-derived from N: the output gate appends W3's rows to B and pads to a work-tile
        # edge, so `B.shape[-2]` is the only spelling that cannot drift from `append_gate3_weight`.
        B.shape[-2],
        PostAct.shape[-1],
        # The DUAL pre-activation's width, BAKED. Taken from whichever tensor actually carries it:
        # `mColsum` is UNCONDITIONAL on the fold path (the fold always emits a column sum) while the
        # interleaved bias is absent whenever the caller passed no projection bias, so deriving this
        # from the bias alone yields 0 there and `(?, 0)` is not a legal layout -- an MLIR parse
        # error naming a shape mode, three frames into tracing. `2 * n` is the fallback rather than
        # 0 for the same reason: the value is unused when neither tensor exists, and a positive one
        # cannot become an invalid extent if a later path does read it.
        (2 * n if rowvec_bias is None else rowvec_bias.shape[-1]),
        n_dual_tiles,
        gate3_n3,
        device_capacity,
    )

    from fold_cp_ops._internal.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    epi_args = DualGatedGemmSm90.EpilogueArguments(
        mPostAct=PostAct_p,
        act_fn=None,  # Constexpr: baked at compile, passed as None at launch
        mRowVecBroadcast=rowvec_bias,
        mMaskColVec=mask_p,
        mBiasUp=half_up,
        mBiasGate=half_gate,
        mPostAct3=PostAct3_p,
        act_fn_3=None,  # Constexpr: baked at compile
        mRowVecBroadcast3=b3_p,
        rounding_mode=None,  # Constexpr: baked at compile
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, max_swizzle_size, tile_count_semaphore
    )
    if two_tensor_B:
        compiled_fn(A_p, B_p, None, None, epi_args, scheduler_args, B2_p)
    else:
        compiled_fn(A_p, B_p, None, None, epi_args, scheduler_args)


def dual_gated_gemm_ref(
    A: Tensor,
    Wg: Tensor,
    Wp: Tensor,
    bg: Optional[Tensor] = None,
    bp: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    W3: Optional[Tensor] = None,
    b3: Optional[Tensor] = None,
    activation: str = "glu",
    transpose_out: bool = False,
):
    """Reference implementation in fp32 torch, for tests and for reading the contract.

    Purpose
        States the operation once, in a form with no tiling, no layout and no fused epilogue, so a
        kernel disagreement localizes to the kernel.

    Semantics
        Everything is computed in fp32 regardless of the inputs' dtype, and the result is returned
        in fp32 -- narrowing is the caller's decision, because the kernel's own narrowing is a
        separate step a test may want to model differently. The mask is applied AFTER the gate, in
        fp32, which is where the kernel applies it.

    Args:
        A: ``(m, k)`` activation.
        Wg: ``(n, k)`` gate weight.
        Wp: ``(n, k)`` up weight.
        bg: Optional ``(n,)`` gate bias.
        bp: Optional ``(n,)`` up bias.
        mask: Optional ``(m,)`` post-gate multiplier.
        W3: Optional ``(n3, k)`` output-gate weight. When given, a SECOND result is returned.
        b3: Optional ``(n3,)`` output-gate bias.
        activation: Gate name; a key of `gate_fn_map`. Only ``"glu"`` is modelled here, matching
            what the kernel ships.
        transpose_out: Return the dual result as ``(n, m)`` instead of ``(m, n)``. The output
            gate's result is ALWAYS ``(m, n3)``, matching the kernel.

    Returns:
        The fp32 dual result, or ``(dual, gate3)`` when `W3` is given.

    Raises:
        ValueError: If `activation` is not modelled here.
    """
    if activation != "glu":
        raise ValueError(f"the reference models only 'glu'; got {activation!r}")
    a32 = A.float()
    gate = a32 @ Wg.float().T
    up = a32 @ Wp.float().T
    if bg is not None:
        gate = gate + bg.float()
    if bp is not None:
        up = up + bp.float()
    out = torch.sigmoid(gate) * up
    if mask is not None:
        out = out * mask.float().reshape(-1, 1)
    out = out.T.contiguous() if transpose_out else out
    if W3 is None:
        return out
    g3 = a32 @ W3.float().T
    if b3 is not None:
        g3 = g3 + b3.float()
    return out, torch.sigmoid(g3)
