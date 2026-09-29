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
"""The dual-gated epilogue: one GEMM twice as wide, halved by a gate.

A dual-gated projection computes two linear maps of the same activation and combines them
pointwise::

    gate[m, n] = (A @ Wg^T)[m, n] + bg[n]
    up  [m, n] = (A @ Wp^T)[m, n] + bp[n]
    out [m, n] = sigmoid(gate[m, n]) * up[m, n]

Running that as two GEMMs would read A twice. Instead it is ONE GEMM against a 2N-wide weight
holding both projections, whose epilogue folds the 2N pre-activation down to an N-wide output. The
mainloop is unchanged -- this module is only the fold.

**Two weight layouts, and which register pairs with which.** How the two projections are laid out
across the 2N axis decides which two accumulator registers the gate combines, and getting it wrong
produces a plausible wrong answer rather than an error. `chunk_g` names the layout:

* ``chunk_g == 1`` -- element interleave. Column ``2i`` is gate ``i``, ``2i+1`` is up ``i``, so the
  pair is ADJACENT registers. Cheap in the epilogue, but the host must interleave the two weight
  matrices, which is a kernel launch on every call.
* ``chunk_g >= 16`` -- block interleave. The 2N axis is ``[up_G | gate_G]`` blocks of width G, and
  the epilogue N-subtile is forced to exactly ``2G`` so one block is register-local: the up half is
  the first ``H = size/2`` registers and the gate half the last ``H``, with up register ``j``
  pairing with gate register ``j+H`` on the SAME thread. No host interleave -- the two weights are
  loaded directly by a two-tensor TMA -- which is the preferred layout when the weights are not
  pre-transformed.

**The gate halves the tile, and that is what makes this more than arithmetic.** The output tile is
half the pre-activation's width, so the post-activation store needs its own epilogue tile
(`gated_epi_tile_fn`), its own ``stmatrix`` matrix count (a ``tile_N`` that is a multiple of 16 but
not 32 leaves an 8-wide post-activation tile), and -- in the element-interleave layout only -- a
register permute, because compressing a column pair leaves the survivor owned by the wrong lane.

**Two seams, so the LayerNorm-fused variants do not have to fork this.** Both known fusions produce
a 2N pre-activation and then need EXACTLY this fold; what differs is only how the pre-activation is
formed:

* the physical-normalization fusion normalizes A in shared memory before the MMA, so its
  pre-activation is already correct and it changes nothing here;
* the algebraic-correction fusion never materializes the normalized A -- it applies a rank-one
  correction ``r*acc - s*c + d`` to the raw accumulator in the epilogue instead.

`epi_combine_preact` is the seam for the second: override it to form the pre-activation however you
like. `epi_gate_preact` is the fold itself and is meant to be INHERITED, not overridden -- it is the
part that was duplicated verbatim in the upstream fusions, and duplicating it again is the specific
regression these seams exist to prevent.
"""

from typing import Callable, NamedTuple, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from fold_cp_ops._internal.epi_act import GemmActMixin
from fold_cp_ops._internal.epi_default import GemmDefaultEpiMixin
from fold_cp_ops._internal.epi_ops import ChunkedHalfBiasLoad, ColVecLoad, TileStore
from fold_cp_ops._internal.reg_permute import permute_gated_Cregs_b16
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops._internal.runtime_params import ParamsBase, mlir_namedtuple


def gated_epi_tile_fn(gemm, epi_tile):
    """Halve the N extent of the epilogue tile for the gated post-activation store.

    Purpose
        The gate turns a 2N-wide pre-activation subtile into an N-wide output, so the
        post-activation's staging tile, TMA box and store atom must all be half as wide as D's.
        `TileStore` calls this to derive that tile rather than inheriting D's.

    Args:
        gemm: The kernel functor. Unused -- the signature is `TileStore`'s hook contract, which
            passes it so a store can depend on kernel configuration.
        epi_tile: D's epilogue tile ``(M, N)``. Its N must be even; an odd N would silently floor,
            producing a store tile that does not cover the fold's output.

    Returns:
        The same tile with N halved. A `cute.Layout` N is halved with `recast_layout` so the
        strides stay consistent; a plain int is divided.
    """
    if isinstance(epi_tile[1], cute.Layout):
        return (epi_tile[0], cute.recast_layout(2, 1, epi_tile[1]))
    return (epi_tile[0], epi_tile[1] // 2)


class GemmGatedMixin(GemmActMixin):
    """Folds a 2N-wide pre-activation to an N-wide output with a pointwise gate.

    Extends `GemmActMixin`, so the post-activation store, its mask and its dtype conversion are
    inherited; what this adds is the fold and the halved tile geometry it implies.

    Requirements beyond `GemmActMixin`'s:

    - **``chunk_g`` is 1 or a multiple of 16.** Asserted in :meth:`maybe_override_epi_tile`. 16 is
      the ``stmatrix`` atom's N-width, so a block that is not a multiple of it would put the up/gate
      split INSIDE an atom, where the register half-split does not exist. ``chunk_g == 8`` is the
      tempting value and it is exactly the unsupported one.
    - **``tile_N`` is a multiple of 16.** Asserted. Not a WGMMA limit -- the MMA atom is legal at
      N%8 and the base `GemmSm90` accepts %16 -- but the gate halves the epilogue N-subtile, and
      below 16 the halved tile has no valid ``stmatrix`` form.
    - **``tile_N`` is a multiple of ``2*chunk_g``** when ``chunk_g > 1``. Asserted. Otherwise a
      ``[up_G | gate_G]`` block straddles two epilogue subtiles and the half-split pairs registers
      from different columns.
    - **D, if written, must be n-major.** Asserted. D holds the 2N pre-activation, whose layout the
      fold's register indexing assumes.
    - **The weight's 2N axis must actually be laid out as ``chunk_g`` says.** NOT checkable here --
      the kernel sees a ``(K, 2N)`` matrix either way. A mismatch pairs gate with the wrong up and
      yields a wrong result with no diagnostic. The front door owns this.
    """

    #: Contiguous up/gate chunk size along the 2N pre-activation. 1 is element interleave; a
    #: multiple of 16 is block interleave. See the module docstring for what each implies.
    chunk_g = 1

    #: The default terms, plus: the post-activation mask (inherited in spirit from `GemmActMixin`
    #: but redeclared because the op ORDER defines the params struct), the two half-width bias
    #: vectors the block-interleave layout loads directly, and the halved post-activation store.
    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        ColVecLoad("mMaskColVec"),
        ChunkedHalfBiasLoad("mBiasUp"),
        ChunkedHalfBiasLoad("mBiasGate"),
        TileStore("mPostAct", epi_tile_fn=gated_epi_tile_fn),
    )
    _extra_param_fields = (("act_fn", cutlass.Constexpr, None),)
    _epi_param_bases = (ParamsBase,)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        """The epilogue terms for a dual-gated GEMM, as passed per launch.

        Attributes:
            mPostAct: ``(M, N)`` 16-bit gated output -- HALF the pre-activation width. May be
                n-major, or m-major for the transposed store.
            act_fn: The gate, a `Constexpr` callable taking ``(gate, up)``. Required in practice: a
                gated GEMM with no gate would store half its accumulator.
            alpha: Scale on the accumulator; see `GemmDefaultEpiMixin.EpilogueArguments`.
            beta: Scale on C; likewise.
            mRowVecBroadcast: ``(2N,)`` INTERLEAVED ``[bg, bp]`` bias added to the pre-activation.
                Its column order must match `chunk_g`. Mutually exclusive with the two half-width
                vectors below -- supplying both double-counts the bias.
            mColVecBroadcast: ``(l, m)`` additive pre-activation bias, or None.
            mMaskColVec: ``(M,)`` multiplicative POST-gate mask, or None. Applied in fp32 before the
                narrowing store, which is the only placement matching an fp32 reference.
            mBiasUp: ``(N,)`` up-projection bias for the block-interleave layout, added directly to
                the up register half -- no host interleave. None unless ``chunk_g > 1``.
            mBiasGate: ``(N,)`` gate-projection bias, likewise. Must be present exactly when
                `mBiasUp` is.
            rounding_mode: `Constexpr` rounding. Must be `RoundingMode.RN`.
            sr_seed: Stochastic-rounding seed; unused under RN.

        Note:
            The `Constexpr` fields are erased from the ABI, so at LAUNCH they must be passed as
            None.
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
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    def maybe_override_epi_tile(self, epi_tile):
        """Force the epilogue N-subtile to hold exactly one ``[up_G | gate_G]`` block.

        Purpose
            In the block-interleave layout the fold pairs up register ``j`` with gate register
            ``j+H``, which is only true when one whole block sits inside one epilogue subtile. This
            is where that becomes a fact rather than an assumption.

        Semantics
            Returns `epi_tile` unchanged for ``chunk_g == 1`` (element interleave pairs adjacent
            registers and needs no particular subtile width). For ``chunk_g > 1`` it returns an N of
            exactly ``2*chunk_g``. Called at compile time from `_setup_attributes`.

        Args:
            epi_tile: The epilogue tile ``(M, N)`` the base class derived.

        Returns:
            The tile to use, with N possibly replaced by ``2*chunk_g``.

        Raises:
            AssertionError: If ``chunk_g`` is neither 1 nor a multiple of 16, or if ``tile_N`` is
                not a multiple of ``2*chunk_g``. Both are wrong-answer conditions, not slow ones.
        """
        if const_expr(self.chunk_g == 1):
            return epi_tile
        assert self.chunk_g % 16 == 0, (
            f"chunk_g must be 1 (element interleave) or a multiple of 16 (block interleave); got "
            f"{self.chunk_g}. 16 is the stmatrix atom N-width -- a smaller block would put the "
            f"up/gate split inside an atom, where the register half-split does not exist."
        )
        epi_n = 2 * self.chunk_g
        # `tile_shape_mn[1]`, not `cta_tile_shape_mnk[1]`: they are the same tile_N, but this hook
        # runs BEFORE the call parameters are bound (the epilogue tile is one of them), and
        # `cta_tile_shape_mnk` is derived from those. The construction parameter is the one that
        # exists here, and is the more honest source anyway -- tile_N is chosen at construction.
        assert self.tile_shape_mn[1] % epi_n == 0, (
            f"tile_N (={self.tile_shape_mn[1]}) must be a multiple of 2*chunk_g (={epi_n}), "
            f"or an [up|gate] block straddles two epilogue subtiles and the fold pairs registers "
            f"from different columns."
        )
        return (epi_tile[0], epi_n)

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        """Lower the launch-time `EpilogueArguments` into the traced `EpilogueParams`.

        Beyond `GemmActMixin`'s version this halves the post-activation CTA tile -- the fold's whole
        geometric consequence -- and enforces the gated tile-N floor.

        Args:
            args: This launch's `EpilogueArguments`.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            An `EpilogueParams` holding one entry per present term, plus `act_fn`.

        Raises:
            AssertionError: If D is written but not n-major, if ``tile_N`` is not a multiple of 16,
                or for any reason `GemmActMixin._latch_postact_attributes` raises.
        """
        return self.EpilogueParams(**self.gated_params_dict(args))

    def gated_params_dict(self, args):
        """Check the gated geometry, latch the post-activation attributes, and build the params dict.

        Purpose
            The half of :meth:`epi_to_underlying_arguments` a SUBCLASS with extra params still
            needs. A fused-LayerNorm variant adds one field (``eps``) to an otherwise identical
            struct; without this seam it would have to restate the assertions and the tile halving
            to do so, and a restated invariant is one that can drift.

        Semantics
            Side-effecting on purpose: it latches ``postact_dtype``/``postact_layout`` and the
            halved post-activation CTA tile onto the functor, which the store setup reads. Call it
            exactly once per launch, before constructing the params.

        Args:
            args: This launch's `EpilogueArguments`.

        Returns:
            A dict with one entry per declared op plus ``act_fn`` -- ready to be extended and
            splatted into ``self.EpilogueParams``.

        Raises:
            AssertionError: If D is written but not n-major, if ``tile_N`` is not a multiple of 16,
                or for any reason `GemmActMixin._latch_postact_attributes` raises.
        """
        assert self.d_layout is None or self.d_layout.is_n_major_c(), (
            "the gated epilogue's D output holds the 2N pre-activation and must be n-major"
        )
        assert self.cta_tile_shape_mnk[1] % 16 == 0, (
            f"the gated epilogue requires tile_N (={self.cta_tile_shape_mnk[1]}) to be a multiple "
            f"of 16: the gate halves the epilogue N-subtile, and below 16 the halved tile has no "
            f"valid stmatrix form. This is a store-atom floor, not a WGMMA limit."
        )
        self._latch_postact_attributes(args)
        # The fold's geometry: the output tile is half the pre-activation's N.
        self.cta_tile_shape_postact_mn = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1] // 2,
        )
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        return d

    @cute.jit
    def epi_combine_preact(
        self,
        params,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> None:
        """Form the 2N pre-activation in ``tRS_rD``, before the gate folds it.

        Purpose
            **This is the seam a LayerNorm-fused variant overrides.** Everything downstream of it --
            the fold, the mask, the conversion, the permute, the store -- is layout work that no
            fusion needs to change, so a fusion that replaces only this method inherits all of it
            instead of forking the epilogue.

        Semantics
            The default runs `GemmDefaultEpiMixin.epi_visit_subtile` (alpha, C with beta, then the
            broadcast biases) and, in the block-interleave layout, adds the two half-width bias
            vectors directly into their register halves. Both write `tRS_rD` in place.

            An overriding fusion must leave `tRS_rD` holding the FULL 2N pre-activation in fp32,
            in the column order `chunk_g` implies. It may ignore the default terms entirely -- the
            algebraic-correction fusion, for instance, applies ``r*acc - s*c + d`` to the raw
            accumulator and adds its own bias, because its correction has to precede any bias.

        Args:
            params: This kernel's `EpilogueParams`.
            epi_loop_tensors: This subtile's loaded terms, keyed by `_epi_ops` name.
            tRS_rD: The accumulator fragment, updated in place to the pre-activation.
            tRS_rC: The C fragment, or None.

        Returns:
            None; `tRS_rD` is modified in place.
        """
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        if const_expr(self.chunk_g > 1):
            # Direct two-vector bias: up register j takes bp, gate register j+H takes bg, where the
            # two share post-activation column j. `ChunkedHalfBiasLoad` gives the up/gate axis
            # stride 0, so tDr_bg[j] == tDr_bg[j+H]; indexing j+H just reads the gate register's own
            # loaded value. Equivalent to the interleaved (2N,) row-vector add, without the host
            # interleave launch that would precede it.
            tDr_bp = epi_loop_tensors["mBiasUp"]
            tDr_bg = epi_loop_tensors["mBiasGate"]
            if const_expr(tDr_bp is not None):
                H = const_expr(cute.size(tRS_rD) // 2)
                for j in cutlass.range_constexpr(H):
                    tRS_rD[j] += tDr_bp[j]
                    tRS_rD[j + H] += tDr_bg[j + H]

    @cute.jit
    def epi_gate_preact(
        self,
        params,
        tRS_rD: cute.Tensor,
    ) -> cute.Tensor:
        """Fold the 2N pre-activation to an N-wide output with the gate.

        Purpose
            The fold itself, and the reason the two weight layouts exist. **Inherit this; do not
            override it.** It was duplicated verbatim in both upstream LayerNorm fusions, and that
            duplication is what the `epi_combine_preact` seam exists to make unnecessary.

        Semantics
            Reads `tRS_rD` and returns a NEW fp32 fragment of half its size; `tRS_rD` is untouched,
            so the D store still sees the pre-activation. Which two registers pair depends on
            `chunk_g`, resolved at compile time:

            * ``chunk_g == 1``: adjacent pair -- ``out[i] = act(rD[2i], rD[2i+1])``, gate first.
            * ``chunk_g > 1``: register half-split -- ``out[j] = act(rD[j+H], rD[j])`` with
              ``H = size/2``, so the gate comes from the second half and the up from the first.
              Valid only because `maybe_override_epi_tile` forced one block per subtile.

        Args:
            params: This kernel's `EpilogueParams`; ``params.act_fn`` is the gate, called as
                ``act_fn(gate, up)``. Argument order is NOT symmetric.
            tRS_rD: The 2N pre-activation fragment. Its size must be even; an odd size would drop
                the last column silently.

        Returns:
            A new fp32 register fragment of half the input's size, holding the gated output.
        """
        tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        # Materializing the shape (rather than deriving it lazily) keeps the compiler from spilling
        # the fragment to local memory and reloading it.
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rPostAct_layout.shape, self.acc_dtype)
        if const_expr(self.chunk_g == 1):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
        else:
            H = const_expr(cute.size(tRS_rD) // 2)
            for j in cutlass.range_constexpr(H):
                tRS_rPostAct[j] = params.act_fn(tRS_rD[j + H], tRS_rD[j])
        return tRS_rPostAct

    @cute.jit
    def epi_apply_postact_mask(self, epi_loop_tensors, tRS_rPostAct) -> None:
        """Multiply the gated fp32 output by the per-row mask, in place.

        Overrides `GemmActMixin`'s 1:1 version because the fold breaks the index correspondence: the
        mask is loaded over the 2N PRE-activation tile, so post-activation element ``i`` must read
        the mask at the pre-activation index its pair came from.

        Semantics
            No-op when no mask was supplied. The mask is constant along N within a row, so for the
            element-interleave layout either half of the pair carries the same value and
            ``tDrMask[2*i]`` is used; for the block-interleave layout post-activation element ``j``
            comes from up register ``j``, whose row is the same, so the index is direct.

        Args:
            epi_loop_tensors: This subtile's loaded terms.
            tRS_rPostAct: The gated fp32 fragment, modified in place.

        Returns:
            None.
        """
        tDrMask = epi_loop_tensors["mMaskColVec"]
        if const_expr(tDrMask is not None):
            if const_expr(self.chunk_g > 1):
                for j in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                    tRS_rPostAct[j] = tRS_rPostAct[j] * tDrMask[j]
            else:
                for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                    tRS_rPostAct[i] = tRS_rPostAct[i] * tDrMask[2 * i]

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
        epi_gate3: cutlass.Constexpr = False,
    ) -> Optional[cute.Tensor]:
        """Form the pre-activation, fold it with the gate, then mask -- in that order.

        The order is the contract: the mask multiplies AFTER the nonlinear gate (masking before
        would compute ``act(mask*x)``), and the gate reads a pre-activation that already carries
        every bias term (adding a bias after the gate would scale it by the gate).

        Args:
            params: This kernel's `EpilogueParams`.
            epi_loop_tensors: This subtile's loaded terms.
            tRS_rD: The accumulator fragment, updated in place to the 2N pre-activation and left
                there for the D store.
            tRS_rC: The C fragment, or None.
            epi_gate3: Whether this is an output-gate pass. Not used by the dual fold; carried for
                the base hook's signature.

        Returns:
            The gated fp32 fragment, half the width of `tRS_rD`.
        """
        self.epi_combine_preact(params, epi_loop_tensors, tRS_rD, tRS_rC)
        tRS_rPostAct = self.epi_gate_preact(params, tRS_rD)
        self.epi_apply_postact_mask(epi_loop_tensors, tRS_rPostAct)
        return tRS_rPostAct

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
        """Narrow the gated fragment, then re-own it for the store when the layout requires.

        Semantics
            The conversion is `GemmActMixin`'s. What is added is the register permute, and it runs
            for the ELEMENT-INTERLEAVE layout only: compressing pre-activation pair ``(2o, 2o+1)``
            into column ``o`` leaves that column held by the lane that owned ``2o``, which is not
            the lane ``stmatrix`` expects. In the block-interleave layout output column ``o`` comes
            from up column ``o``, contiguous and already correctly owned, so the permute is skipped
            -- running it there would scramble valid data with no error.

        Args:
            tRS_rPostAct: The gated fp32 fragment.
            sr_seed: Stochastic-rounding seed; unused under RN.
            tidx: Calling thread index; unused under RN.
            tile_coord_mnkl: Work-tile coordinate; unused under RN.
            num_prev_subtiles: Subtiles already stored; unused under RN.
            epi_idx: Index of this subtile; unused under RN.
            epi_gate3: Whether this is an output-gate pass. When True the fragment maps 1:1 to its
                output with no fold, so the permute must be skipped.

        Returns:
            A new register fragment of ``self.postact_dtype``, permuted where required.
        """
        tRS_rPostAct_out = GemmActMixin.epi_convert_postact(
            self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
        )
        if const_expr(epi_gate3):
            return tRS_rPostAct_out
        if const_expr(self.chunk_g == 1):
            permute_gated_Cregs_b16(tRS_rPostAct_out)
        return tRS_rPostAct_out
