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
"""The post-activation epilogue: a SECOND output tensor, written through its own TMA store.

`GemmDefaultEpiMixin` produces one output, D. This adds a second, `mPostAct`, carrying an activated
form of the same accumulator, and everything that second store needs: its own SMEM staging buffer,
its own TMA atom, its own register-to-SMEM atom sized to its own tile, and its own dtype conversion.
D remains optional and independent -- a kernel may write both, or only the post-activation.

**Why a second tensor rather than an in-place activation.** The activation is applied to the fp32
accumulator, and callers want *both* the linear result and the activated one often enough that
recomputing the GEMM would dominate. More importantly for the gated subclass, the post-activation
tile is a DIFFERENT SHAPE from D (a gate halves it), so it cannot share D's staging buffer or its
store atom. Splitting the tensor is what makes the shape split expressible.

**The one non-obvious piece: the mask is applied POST-activation.** `mMaskColVec` is a per-row
multiplier, and for a nonlinear activation ``mask * act(x) != act(mask * x)``, so it cannot be
folded into the pre-activation bias terms that `GemmDefaultEpiMixin` already handles. It is applied
to the fp32 result *before* the narrowing store, which is what makes it match a reference that masks
in fp32. Applying it after the store would quantize first and give a different answer.

This mixin is deliberately activation-agnostic: it stores whatever `epi_visit_subtile` hands back.
`epi_gated.GemmGatedMixin` subclasses it to supply a gate, which is the only form the TriMul path
uses; the elementwise form here is what a future output-gate pass plugs into.
"""

from typing import Callable, NamedTuple, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

import fold_cp_ops._internal.copy_utils as copy_utils
from fold_cp_ops._internal.epi_default import GemmDefaultEpiMixin
from fold_cp_ops._internal.epi_ops import ColVecLoad, TileStore
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops._internal.runtime_params import ParamsBase, mlir_namedtuple


class GemmActMixin(GemmDefaultEpiMixin):
    """Adds a post-activation output to the default epilogue.

    Mixed in ahead of a ``GemmSm90``-family base, exactly like `GemmDefaultEpiMixin`, whose terms it
    keeps and extends. The added declarations are two `_epi_ops` -- a column-vector mask load and the
    `mPostAct` tile store -- plus one `Constexpr` parameter, `act_fn`.

    Requirements on what this mixin adds:

    - **``mPostAct`` must be 16-bit** (bf16/fp16). Asserted. The register-to-SMEM atom chosen here is
      an ``stmatrix`` form that only exists for 16-bit types; a 32-bit post-activation would need a
      different store leg entirely.
    - **``mPostAct`` may be n-major or m-major.** Both are handled (m-major is the transposed-output
      layout a downstream batched GEMM wants). Asserted, because a mode-major layout that is neither
      would partition wrongly rather than fail.
    - **``act_fn`` is a ``Constexpr``**, so it is folded in at compile time and forms part of the
      cache key. It must be passed as None at LAUNCH -- see `EpilogueArguments`.
    - **``rounding_mode`` must be ``RoundingMode.RN``.** Stochastic rounding is a Blackwell epilogue
      and this package is SM90-only; the check is here rather than at the front door so a
      programmatic caller building `EpilogueArguments` directly cannot slip past it.
    - **``mMaskColVec``, if present, is a per-row ``(M,)`` vector** broadcast along N. It multiplies
      the post-activation result in fp32. This is NOT the same term as `mColVecBroadcast`, which is
      an additive PRE-activation bias; supplying one where the other is meant is silently wrong.
    """

    #: The default epilogue's terms, plus the two this mixin owns. `mMaskColVec` is a multiplicative
    #: post-activation mask (distinct from `mColVecBroadcast`, an additive pre-activation bias);
    #: `mPostAct` is the second output tile.
    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        ColVecLoad("mMaskColVec"),
        TileStore("mPostAct"),
    )
    #: `act_fn` is a traced constant, not a loaded tensor, so it cannot be an `_epi_ops` entry;
    #: `ComposableEpiMixin` splices declared extra fields into the generated `EpilogueParams`.
    _extra_param_fields = (("act_fn", cutlass.Constexpr, None),)
    _epi_param_bases = (ParamsBase,)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        """The epilogue terms for a post-activation GEMM, as passed per launch.

        Attributes:
            mPostAct: ``(M, N)`` 16-bit post-activation output. Required -- this mixin exists to
                write it. May be n-major or m-major.
            act_fn: The activation, a `Constexpr` callable from `_internal.activation`. None stores
                the raw accumulator, narrowed.
            alpha: Scale on the accumulator; see `GemmDefaultEpiMixin.EpilogueArguments`.
            beta: Scale on C; likewise.
            mRowVecBroadcast: ``(l, n)`` additive pre-activation bias, or None.
            mColVecBroadcast: ``(l, m)`` additive pre-activation bias, or None.
            mMaskColVec: ``(M,)`` multiplicative POST-activation mask, or None. Multiplies in fp32
                before the narrowing store.
            rounding_mode: `Constexpr` rounding for both stores. Must be `RoundingMode.RN` here.
            sr_seed: Stochastic-rounding seed; unused under RN, carried for struct compatibility.

        Note:
            The `Constexpr` fields (`act_fn`, `rounding_mode`) are baked in at compile time and
            their ABI slots are erased, so at LAUNCH they must be passed as None. Passing the real
            value again is an argument-count mismatch, not an override.
        """

        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mMaskColVec: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams is auto-generated from _epi_ops + _extra_param_fields.

    def _latch_postact_attributes(self, args) -> None:
        """Read the post-activation tensor's type and layout off the arguments onto ``self``.

        Purpose
            The store leg needs the post-activation dtype, layout and CTA tile at trace time, and
            they are properties of the tensor rather than of the constructor -- this is the
            two-phase configuration `GemmSm90` documents. Factored out of
            `epi_to_underlying_arguments` because the gated subclass latches the same four
            attributes but with a HALVED N, and duplicating the block is how the two drift apart.

        Semantics
            Sets ``self.postact_dtype``, ``self.postact_layout`` and
            ``self.cta_tile_shape_postact_mn``. The last is the full CTA ``(M, N)`` here; a gate
            overrides it. Validates the requirements listed in the class docstring -- including the
            rounding mode, which is bound as a call-phase template parameter by
            ``bind_operand_types`` and is therefore only CHECKED here, not assigned.

        Args:
            args: This launch's `EpilogueArguments`. ``args.mPostAct`` must be non-None and 16-bit.

        Returns:
            None; mutates ``self``.

        Raises:
            AssertionError: If the post-activation is not 16-bit, is neither n-major nor m-major, or
                the rounding mode is not RN.
        """
        assert args.mPostAct.element_type.width == 16, (
            "the post-activation output must be 16-bit (bf16/fp16); the stmatrix store leg has no "
            "32-bit form"
        )
        pa_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        assert pa_layout.is_n_major_c() or pa_layout.is_m_major_c(), (
            "the post-activation output must be n-major (M, N) or m-major (the transposed store)"
        )
        assert args.rounding_mode == RoundingMode.RN, (
            "stochastic rounding is a Blackwell epilogue; this package is SM90-only, so the "
            "post-activation store supports RoundingMode.RN only"
        )
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = pa_layout
        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        """Lower the launch-time `EpilogueArguments` into the traced `EpilogueParams`.

        Args:
            args: This launch's `EpilogueArguments`.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            An `EpilogueParams` holding one entry per present term, plus `act_fn`.

        Raises:
            AssertionError: See :meth:`_latch_postact_attributes`.
        """
        self._latch_postact_attributes(args)
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        return self.EpilogueParams(**d)

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
        """Build the post-activation store's copies and partitions, once per work tile.

        Purpose
            Everything the per-subtile loop needs to move the activated fragment out: a
            register-to-SMEM tiled copy, this thread's slice of the SMEM staging tile, and the
            SMEM-to-GMEM TMA closure.

        Semantics
            Runs once per work tile, before the epilogue subtile loop. The register-to-SMEM atom is
            chosen from the POST-ACTIVATION tile's own major-mode width rather than D's -- see
            `copy_utils.sm90_get_smem_store_atom` for why that distinction is load-bearing when a
            gate halves the tile. The returned closure is a partitioned TMA store, not a copy: it is
            called per subtile with the staging index.

        Args:
            params: This kernel's `EpilogueParams`.
            epi_smem_tensors: The epilogue's SMEM tensors, indexed through ``self._epi_smem_map``.
            tiled_copy_r2s: D's register-to-SMEM tiled copy, used only as the source layout the
                post-activation copy is built against, so the two stay thread-consistent.
            tiled_copy_t2r: D's SMEM-to-register tiled copy. Unused on SM90; taken so the signature
                matches the base hook.
            tile_coord_mnkl: This work tile's coordinate; its ``[3]`` selects the batch.
            tidx: The calling thread's index within the CTA.
            epi_gate3: Whether this is an output-gate pass. Unused here; a subclass reads it.

        Returns:
            ``(tiled_copy_postact_r2s, tRS_sPostAct, copy_postact)`` -- the tiled copy, this
            thread's SMEM destination partition, and the TMA store closure.
        """
        sPostAct = epi_smem_tensors[self._epi_smem_map["mPostAct"]]
        copy_atom_postact_r2s = copy_utils.sm90_get_smem_store_atom(
            self.postact_dtype,
            transpose=self.postact_layout.is_m_major_c(),
            major_mode_size=params.epi_tile_mPostAct[1],
        )
        tiled_copy_postact_r2s = cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_r2s)
        tRS_sPostAct = tiled_copy_postact_r2s.get_slice(tidx).partition_D(sPostAct)
        batch_idx = tile_coord_mnkl[3]
        copy_postact, _, _ = self.epilog_gmem_copy_and_partition(
            params.tma_atom_mPostAct,
            self.select_batch(params.mPostAct, batch_idx),
            self.cta_tile_shape_postact_mn,
            params.epi_tile_mPostAct,
            sPostAct,
            tile_coord_mnkl,
        )
        return tiled_copy_postact_r2s, tRS_sPostAct, copy_postact

    @cute.jit
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
        """Narrow the fp32 post-activation fragment to the output dtype.

        Semantics
            A plain round-to-nearest conversion into a fresh register fragment; the fp32 source is
            left intact. Round-to-nearest is the only mode: stochastic rounding needs Blackwell and
            is refused in :meth:`_latch_postact_attributes`, so there is no mode branch here.

        Args:
            tRS_rPostAct: The fp32 fragment for this subtile.
            sr_seed: Stochastic-rounding seed; unused under RN.
            tidx: Calling thread index; unused under RN.
            tile_coord_mnkl: Work-tile coordinate; unused under RN.
            num_prev_subtiles: Subtiles already stored; unused under RN.
            epi_idx: Index of this subtile; unused under RN.
            epi_gate3: Whether this is an output-gate pass. Unused here.

        Returns:
            A new register fragment of ``self.postact_dtype`` holding the narrowed values.
        """
        tRS_rPostAct_out = cute.make_rmem_tensor_like(tRS_rPostAct, self.postact_dtype)
        tRS_rPostAct_out.store(tRS_rPostAct.load().to(self.postact_dtype))
        return tRS_rPostAct_out

    @cute.jit
    def epi_apply_postact_mask(self, epi_loop_tensors, tRS_rPostAct) -> None:
        """Multiply the fp32 post-activation by the per-row mask, in place.

        Purpose
            Applies `mMaskColVec` AFTER the activation and BEFORE the narrowing store, which is the
            only placement that matches a reference masking in fp32: the activation is nonlinear, so
            ``mask * act(x) != act(mask * x)``, and masking after the store would quantize first.

        Semantics
            No-op -- no instructions emitted -- when no mask was supplied. The mask is a column
            vector broadcast along N, so it is constant within a row and shares the PRE-activation
            fragment's index space. This base implementation is the 1:1 case, where post-activation
            element ``i`` corresponds to pre-activation element ``i``; a gate compresses two
            pre-activation columns into one and must override the indexing.

        Args:
            epi_loop_tensors: This subtile's loaded terms, keyed by `_epi_ops` name.
            tRS_rPostAct: The fp32 post-activation fragment, modified in place.

        Returns:
            None.
        """
        tDrMask = epi_loop_tensors["mMaskColVec"]
        if const_expr(tDrMask is not None):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = tRS_rPostAct[i] * tDrMask[i]

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
        epi_gate3: cutlass.Constexpr = False,
    ) -> Optional[cute.Tensor]:
        """Combine the default terms, then apply the activation elementwise.

        Semantics
            Runs `GemmDefaultEpiMixin.epi_visit_subtile` first, so `tRS_rD` becomes the
            pre-activation (alpha-scaled accumulator plus C and the broadcast biases) and is what
            the D store will write. The activation then produces a SEPARATE fragment, so both
            outputs stay available. With no activation the post-activation fragment aliases `tRS_rD`
            rather than being copied -- writing to the returned fragment in that case also modifies
            D.

        Args:
            params: This kernel's `EpilogueParams`.
            epi_loop_tensors: This subtile's loaded terms.
            tRS_rD: The accumulator fragment, updated in place to the pre-activation.
            tRS_rC: The C fragment, or None.
            epi_gate3: Whether this is an output-gate pass. Unused here.

        Returns:
            The fp32 post-activation fragment for this subtile.
        """
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        if const_expr(params.act_fn is not None):
            tRS_rPostAct = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = params.act_fn(tRS_rD[i])
        else:
            tRS_rPostAct = tRS_rD
        self.epi_apply_postact_mask(epi_loop_tensors, tRS_rPostAct)
        return tRS_rPostAct
