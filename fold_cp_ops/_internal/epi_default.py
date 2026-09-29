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
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from fold_cp_ops._internal.runtime_params import mlir_namedtuple
from fold_cp_ops._internal.epi_composable import ComposableEpiMixin
from fold_cp_ops._internal.epi_ops import Scalar, RowVecLoad, ColVecLoad
from fold_cp_ops._internal.rounding import RoundingMode
import fold_cp_ops._internal.utils as utils


class GemmDefaultEpiMixin(ComposableEpiMixin):
    """The default GEMM epilogue: ``D = alpha * acc + beta * C + rowvec + colvec``.

    Mixed in *before* a ``GemmSm90``-family base, whose no-op ``epi_*`` hooks it overrides. Every
    term is optional at the argument level and each one that is absent is compiled out entirely --
    the epilogue does not multiply by one or add zero, the instructions are simply not emitted --
    which is why the presence of each term is part of the compile cache key rather than a runtime
    input.

    ``_epi_ops`` declares the terms; ``ComposableEpiMixin`` reads that declaration and generates the
    ``EpilogueParams`` struct, the SMEM/TMA plumbing, and the per-subtile load hooks from it. This
    class supplies only what the composition cannot infer: the argument NamedTuple and the arithmetic
    that combines the loaded terms.

    Requirements on the arguments this mixin adds (all carried in ``EpilogueArguments``, not in the
    constructor): a scalar may be a value or a device pointer but the CHOICE is compile-time; the
    broadcast vectors must have a 4-element-aligned row pitch, which for a column vector means M;
    and ``rounding_mode`` must be ``RoundingMode.RN`` on SM90, since stochastic rounding is a
    Blackwell epilogue.
    """

    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecLoad("mColVecBroadcast"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        """The epilogue terms, as passed per launch.

        Attributes:
            alpha: Scale on the accumulator. None folds the multiply away; a ``Float32`` bakes the
                constant in; a ``cute.Tensor``/pointer reads it at launch. The three are different
                compiled kernels.
            beta: Scale on C, with the same three modes. Ignored when the kernel was compiled
                without a C operand.
            mRowVecBroadcast: ``(l, n)`` vector added to every row, or None.
            mColVecBroadcast: ``(l, m)`` -- added to every
                column, or None. The RANK is part of the compile key, so the two forms differ.
            add_to_output: Compile-time flag making the store accumulate into D instead of
                overwriting it.
            rounding_mode: Compile-time ``RoundingMode``. Read by ``epi_to_underlying_arguments``
                into ``self.rounding_mode``, which the shared epilogue consults when narrowing.
            sr_seed: Stochastic-rounding seed, unused under ``RoundingMode.RN``.

        Note:
            The ``Constexpr`` fields are baked in at compile time and their ABI slots are
            ``ConstNone``-erased, so at LAUNCH they must be passed as None. Passing the real value
            again is an argument-count mismatch, not an override.
        """

        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        """Convert the launch-time ``EpilogueArguments`` into the traced ``EpilogueParams``.

        ``self.rounding_mode`` -- the one piece of epilogue configuration the shared
        ``GemmSm90.epilogue()`` reads off the functor rather than off the params -- is NOT latched
        here. It is a call-phase template parameter, bound by ``bind_operand_types`` via
        ``GemmSm90.epi_rounding_mode(args)`` before this runs, and immutable from then on. Assigning
        it here would raise, which is the point: it is folded into the kernel, so a later write
        could only ever desynchronize the functor from the kernel compiled off it.

        Args:
            args: The ``EpilogueArguments`` NamedTuple for this launch.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            An ``EpilogueParams`` built from the declared ``_epi_ops``, holding one entry per term
            that is present.
        """
        d = self._epi_ops_to_params_dict(args)
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
        epi_gate3: cutlass.Constexpr = False,
    ) -> Optional[cute.Tensor]:
        """Combine one epilogue subtile: scale, add C, then add the broadcasts.

        The order is load-bearing. alpha scales the accumulator only; C is added *after* that scale
        and carries its own beta; the biases are added last and unscaled. A different order would be
        a different function, not a rounding difference.

        Every term is guarded by a ``const_expr`` on whether it is present, so a kernel compiled
        without (say) a column bias emits no loop for it at all.

        Args:
            params: This kernel's ``EpilogueParams``.
            epi_loop_tensors: The per-subtile loaded values, keyed by the ``_epi_ops`` names.
            tRS_rD: The accumulator fragment for this subtile, **updated in place** and also the
                source of the values being combined.
            tRS_rC: The C fragment for this subtile, or None when the kernel has no C. When present
                and ``beta`` is absent, C is added with an implied beta of 1.0 -- not skipped.
            epi_gate3: Whether this is a fused TriMul output-gate invocation. Unused here.

        Returns:
            None -- the default epilogue has no second (postact) output. A gated subclass returns
            the pre-activation fragment instead.
        """
        alpha = epi_loop_tensors["alpha"]
        beta = epi_loop_tensors["beta"]
        tDrRowVec = epi_loop_tensors["mRowVecBroadcast"]
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        rD = tRS_rD.load()
        # Apply alpha scaling to accumulator if alpha is provided (not None)
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha
        # Apply C with beta scaling
        if const_expr(tRS_rC is not None):
            if const_expr(not hasattr(params, "beta") or params.beta is None):
                # beta is None, default behavior: add C (beta=1.0)
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                beta = utils.load_scalar_or_pointer(params.beta)
                rD += beta * tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)
        if const_expr(tDrRowVec is not None):
            for i in cutlass.range(cute.size(tDrRowVec), unroll_full=True):
                tRS_rD[i] += tDrRowVec[i]
        if const_expr(tDrColVec is not None):
            for i in cutlass.range(cute.size(tDrColVec), unroll_full=True):
                tRS_rD[i] += tDrColVec[i]
        return None

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
        """Returns None — default epilogue has no postact output."""
        return None

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
        """Convert postact from acc_dtype to output dtype. Override for custom postprocessing."""
        return tRS_rPostAct
