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

"""Tests for ``fold_cp_ops.kernels.layernorm_dual_gated_gemm`` -- the fused TriMul input projection.

The reference gates here carry two roundings the unfused kernel's do not (the normalize writes back
in 16 bits, and the gate is nonlinear), so the tolerance is necessarily loose. The gates that
actually pin behaviour are the ones that need NO reference:

* the two weight layouts (``chunk_g`` 1 vs 16) form the same pre-activation from different register
  pairings and must agree BIT FOR BIT;
* the transposed store writes the same values through a different descriptor;
* the fused output gate must not perturb the dual output it shares a mainloop with;
* omitting the LayerNorm bias and passing a zero one must give identical bits.

**On ``blk_k``.** It is swept as an axis and deliberately NOT asserted to be bitwise-invariant. It
sets the k-loop trip count, so it reassociates both the per-row sums and the accumulator; the
functor documents it as a numerical parameter, and ``tests/_internal/test_ln_prologue.py`` asserts
that two widths genuinely differ. Here it is swept to prove every width is CORRECT, not identical.
"""

import warnings
import ast
import inspect
import itertools
import math
import textwrap

import pytest
import torch

from fold_cp_ops._internal.compile_time.ln_prologue_layout import ALLOWED_BLK_K, auto_blk_k
from fold_cp_ops.kernels.dual_gated_gemm import DualGatedGemmParams

# The launch-counting helper, imported rather than copied: it is the SAME measurement the
# unfused operand build is gated on, and two copies of a profiler wrapper would drift.
from tests.kernels.test_dual_gated_gemm import _device_kernels
from fold_cp_ops._internal.heuristic_arch import _WARNED, TUNED_ARCH
from fold_cp_ops.kernels.layernorm_dual_gated_gemm import (
    ALG_FOLD_TUNING_SPACE,
    alg_fold_config_is_valid,
    alg_fold_freeze,
    alg_fold_heuristic_config,
    alg_fold_tuning_space,
    layernorm_dual_gated_gemm_alg_fold,
)
from fold_cp_ops.kernels.layernorm_dual_gated_gemm import (
    FUSION_VARIANTS,
    LayerNormDualGatedGemmAlgFoldSm90,
    LayerNormDualGatedGemmParams,
    LayerNormDualGatedGemmSm90,
    LayerNormDualGatedGemmXGateAlgFoldSm90,
    LayerNormDualGatedGemmXGatePrologLnSm90,
    _XGateTwoASm90,
    _xgate_functor_for,
    build_folded_dual_operands,
    build_folded_dual_operands_ref,
    layernorm_dual_gated_gemm,
    layernorm_dual_gated_gemm_ref,
    resolve_k_tiling,
)
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    dtype_facets,
    front_door_raises,
    matrix_exempt,
)
from fold_cp_ops.testing.numerics import (
    alg_fold_error_bound,
    alg_fold_sigmoid_error_bound,
    alg_fold_xgate_error_bound,
    assert_bitwise,
    assert_elementwise,
    fused_ln_gated_error_bound,
    fused_ln_sigmoid_error_bound,
)

# The hardware facts the four shipped bounds are derived from, borrowed rather than re-derived by
# the two-A physical bound below. See its Note for why it lives here and not beside them.
from fold_cp_ops.testing.numerics import (
    _PRECISION_BITS,
    _SIGMOID_ABS_ERR,
    _SIGMOID_MAX_SLOPE,
    _U_FP32,
)

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(
    _SM != 9, reason=f"layernorm_dual_gated_gemm needs sm_90; this GPU is sm_{_SM}0"
)

LN_DUAL_GATED = KernelMatrix(
    kernel="layernorm_dual_gated_gemm",
    axes=(
        Axis(
            name="ab_dtype",
            domain=(
                "bfloat16 or float16 for the activation, both weights and the gated output -- the "
                "16-bit widths the SM90 WGMMA atom and the stmatrix store leg both have a form "
                "for. float32 is in the pool because it is a KEY of torch2cute_dtype_map (the "
                "LayerNorm gain legitimately IS fp32) and so reaches the entry, where it must be "
                "refused as an OPERAND rather than three frames into tracing"
            ),
            values=(torch.bfloat16, torch.float16, torch.float32),
            facets=dtype_facets((torch.bfloat16, torch.float16, torch.float32)),
        ),
        Axis(
            name="chunk_g",
            domain=(
                "1 (element interleave: the host interleaves the two weights, the epilogue pairs "
                "adjacent registers) or a multiple of 16 (block interleave: the weights load "
                "directly through a two-tensor TMA and the epilogue pairs across a register "
                "half-split). 8 is in the pool because it is the tempting value -- half the "
                "stmatrix atom width -- and must be refused rather than silently mis-paired"
            ),
            values=(1, 8, 16),
            facets={
                "element_interleave": lambda v: v == 1,
                "block_interleave": lambda v: v > 1 and v % 16 == 0,
                "sub_atom": lambda v: 1 < v < 16,
            },
        ),
        Axis(
            name="blk_k",
            domain=(
                "the mainloop K tile: 64, 32 or 16, each a multiple of the 16-bit WGMMA atom's K "
                "extent and each required to divide K exactly. A NUMERICAL parameter, not only a "
                "performance one -- it sets the k-loop trip count, so it decides how the per-row "
                "sums and the accumulator are associated"
            ),
            values=ALLOWED_BLK_K,
            facets={
                "default": lambda v: v == 64,
                "narrow": lambda v: v < 64,
                "narrowest": lambda v: v == 16,
            },
        ),
        Axis(
            name="ln_bias",
            domain=(
                "bool; whether a (K,) LayerNorm bias is supplied. Its ABSENCE is a compile key -- "
                "the add is pruned rather than made zero -- so the two are different kernels, not "
                "one kernel with a zero operand"
            ),
            values=(False, True),
            facets={"affine": lambda v: v, "gain_only": lambda v: not v},
        ),
        Axis(
            name="bias",
            domain=(
                "'none', 'both' (a bias on each projection), or 'gate_only'. The last is in the "
                "pool because biasing one projection is a plausible typo that nothing downstream "
                "can detect -- the pre-activation would simply be half-biased"
            ),
            values=("none", "both", "gate_only"),
            facets={
                "unbiased": lambda v: v == "none",
                "biased": lambda v: v == "both",
                "half_biased": lambda v: v == "gate_only",
            },
        ),
        Axis(
            name="mask",
            domain=(
                "bool; a per-row multiplier applied to the fp32 output AFTER the gate. It cannot "
                "be folded into the projection biases -- the gate is nonlinear -- so it is a "
                "distinct epilogue term, not a scaling of an existing one"
            ),
            values=(False, True),
            facets={"masked": lambda v: v, "unmasked": lambda v: not v},
        ),
        Axis(
            name="transpose_out",
            domain=(
                "bool; whether the gated output is m-major (a transposed view, which the "
                "downstream batched GEMM wants) rather than n-major. Selects a different store "
                "descriptor and a different stmatrix atom, not a different result"
            ),
            values=(False, True),
            facets={"m_major": lambda v: v, "n_major": lambda v: not v},
        ),
        Axis(
            name="gate3",
            domain=(
                "bool; whether the fused TriMul output gate is built. Its weight's rows are "
                "APPENDED to the dual weight, so an output-gate tile is an ordinary work tile of "
                "the same operand -- but the region boundary and the second store are compile-time "
                "facts, so it is a different kernel and not a runtime option"
            ),
            values=(False, True),
            facets={"gated_output": lambda v: v, "dual_only": lambda v: not v},
        ),
        Axis(
            name="x_gate",
            domain=(
                "bool; whether the GATE projection reads a SEPARATE, already-normalized activation "
                "instead of sharing the value's LN(x). True selects a different kernel entirely -- "
                "two A operands, two WGMMA streams, two N-wide accumulators combined in the "
                "epilogue -- so it is a compile-time fact, not a runtime option. It is the TriMul "
                "BACK half's out-gate, where x_gate is the same LN(x) the caller already computed "
                "for another consumer, and it is XOR with gate3: both ask this kernel to compute a "
                "second thing and there is one accumulator pair to do it with"
            ),
            values=(False, True),
            facets={"two_activations": lambda v: v, "one_activation": lambda v: not v},
        ),
        Axis(
            name="fusion_variant",
            domain=(
                "the compile-time fusion selector. 'prolog_ln' is the physical fusion (normalize A "
                "in shared memory, then a plain GEMM, two sweeps over A); 'alg_fold' is the "
                "algebraic rank-one correction (raw x against a pre-folded weight, one sweep). "
                "BOTH are built, and each carries its own error bound -- they are not two spellings "
                "of one kernel, so a correctness cell that runs only one covers only one"
            ),
            values=FUSION_VARIANTS,
            facets={
                "physical": lambda v: v == "prolog_ln",
                "algebraic": lambda v: v == "alg_fold",
            },
        ),
        Axis(
            name="arg_fault",
            domain=(
                "a malformed TENSOR ARGUMENT, or 'none'. Modelled on test_gemm.py's `layout` axis: "
                "the values name a CALLER MISTAKE rather than a kernel configuration, and every "
                "non-'none' one must be refused at the front door. The pool spans the dtype AND the "
                "extent of both an operand and a non-operand tensor on purpose -- which of the two "
                "a tensor is does not appear in this kernel's signature, so it must not decide "
                "whether the mistake is named. Measured before this axis existed: a bad `mask` "
                "dtype raised a bare `KeyError` from the compile-key lookup, which is exactly the "
                "failure class `API_LEVEL_ERRORS` exists to forbid"
            ),
            values=("none", "mask_dtype", "mask_extent", "bias_extent", "postact_extent"),
            facets={
                "well_formed": lambda v: v == "none",
                "bad_dtype": lambda v: v.endswith("_dtype"),
                "bad_extent": lambda v: v.endswith("_extent"),
                "non_operand": lambda v: v.startswith(("mask", "bias")),
            },
        ),
        Axis(
            name="pingpong",
            domain=(
                "bool; two MMA warpgroups alternating mainloop and epilogue instead of sharing one "
                "tile. A COMPILE key and a numerics-relevant one: it makes the row reduction "
                "warpgroup-local, which changes how many rows a thread owns and gives each "
                "warpgroup its own statistics scratch. Only the algebraic fold accepts it -- "
                "prolog_ln's normalize needs the threads that reduced a row to be the ones that "
                "rescale it -- so True with prolog_ln must RAISE rather than hang"
            ),
            values=(False, True),
            facets={
                "cooperative": lambda v: not v,
                "two_warpgroup": lambda v: v,
            },
        ),
        Axis(
            name="tile_M",
            domain=(
                "the CTA tile's M extent, a front-door parameter. It is an AXIS and not a "
                "constant because the number of epilogue subtiles along M is a function of it, and "
                "an op that ignored the subtile coordinate was correct at every value the tests "
                "happened to use and wrong at 256 -- silently, by ~20%. 320 is in the pool and "
                "must RAISE: its rows do not divide the reduction's row-group count. "
                "**192 is deliberately OUT of the pool**, and that is an exclusion rather than an "
                "oversight: whether it is supported depends on tile_N (at or below 128 the base "
                "gives it three warpgroups and it tiles; above, two, and it does not), and tile_N "
                "is not an axis here -- so neither a supported nor an unsupported region could "
                "state the truth about it. The front door refuses the unsupported half; "
                "test_a_tile_that_does_not_divide_the_pre_activation_is_still_correct is where a "
                "tile_N-dependent claim would go"
            ),
            values=(64, 128, 256, 320),
            facets={
                "single_m_subtile": lambda v: v in (64, 128),
                "multi_m_subtile": lambda v: v >= 256,
                "default": lambda v: v == 128,
            },
        ),
        Axis(
            name="M",
            domain=(
                "token extent, entirely free -- no tile multiple, no power of two. The pool spans "
                "the partial-tile case (below one tile_M), the off-grid cases, the MULTI-WAVE "
                "regime the persistent scheduler only reaches past ~132 CTAs' worth of tiles "
                "(which is where a per-CTA scratch reused across tiles is exercised at all), AND "
                "the token counts the A2A-fused TriMul workflow actually runs: its front consumes "
                "a local shard of N_token**2 / cp rows, so at cp=16 the N_token ladder 2048 / "
                "4096 / 8192 is M = 262144 / 1048576 / 4194304. Those three carry the "
                "`workflow_M` facet and are the SAME pool the perf gate builds its front-projection "
                "cells from, so a shape is not correct at one size and timed at another"
            ),
            values=(64, 129, 1000, 4096, 32768, 262144, 1048576, 4194304),
            facets={
                "partial_tile": lambda v: v < 128,
                "tile_multiple": lambda v: v % 128 == 0,
                "off_grid": lambda v: v % 128 != 0,
                "multi_wave": lambda v: v >= 32768,
                # The workflow's own shard sizes. N_token=12288 (M = 9437184) is the one member of
                # that ladder deliberately absent: at D=512 it is 9.7 GB in and 9.7 GB out before
                # any fp32 reference, and the 4.2M cell already makes every point it would.
                # 9437184 (N_token=12288) is DELIBERATELY absent, and the table permits
                # exactly this one omission from a CORRECTNESS pool: at D=512 the cell is
                # 9.7 GB in and 9.7 GB out before any fp32 reference. It stays in the perf
                # cells, which need no oracle. Recorded here so the next reader sees a
                # decision rather than a gap -- which is the whole job of the facet.
                "workflow_M": lambda v: v in (262144, 1048576, 4194304),
            },
        ),
        Axis(
            name="N",
            domain=(
                "output feature extent, i.e. HALF the pre-activation width. Free. The pool carries "
                "the four TriMul feature widths the A2A workflow actually runs (128/256/384/512) "
                "plus a narrow one and an off-grid one, so a pool that covered only the convenient "
                "width cannot satisfy the diversity check -- plus 320, which is the ONLY value "
                "whose largest dividing tile is narrower than 128, and the only one that reaches "
                "the stage-recycle race `narrow_dividing_tile` names"
            ),
            values=(64, 128, 137, 256, 320, 384, 512),
            facets={
                "narrow": lambda v: v <= 64,
                "off_grid": lambda v: v % 128 != 0,
                "workflow_D": lambda v: v in (128, 256, 384, 512),
                # The largest %32 tile dividing N is 64 or 96 rather than 128, so a CTA covers the
                # output width in more, narrower tiles and the epilogue runs more subtiles per work
                # tile. That is what exposed the missing statistics fence (see
                # `test_the_two_activation_stats_survive_a_narrow_dividing_tile`); every other N in
                # the pool either divides by 128 or has no dividing tile at all, which is why this
                # facet was empty and the defect shipped.
                "narrow_dividing_tile": lambda v: v > 256 and v % 128 != 0,
            },
        ),
        Axis(
            name="K",
            domain=(
                "contraction extent, and the ONLY extent carrying a constraint: the 16-byte "
                "alignment floor. The pool is the workflow's feature widths plus a non-power-of-two "
                "that still meets the floor, because K is where a tile-multiple assumption would "
                "hide -- plus 136, which meets the floor and NOTHING else: no member of "
                "ALLOWED_BLK_K divides it, so it is the only value that exercises the padded "
                "contraction at all. The other five are every one a multiple of 64, which is how "
                "the padded path shipped broken"
            ),
            values=(128, 136, 192, 256, 384, 512),
            facets={
                "workflow_D": lambda v: v in (128, 256, 384, 512),
                "non_power_of_two": lambda v: v & (v - 1) != 0,
                "widest": lambda v: v >= 512,
                "no_k_tile_divides": lambda v: all(v % t for t in ALLOWED_BLK_K),
            },
        ),
        Axis(
            name="row_mean",
            domain=(
                "any float, added to every element of `x` on top of the per-row offset `_build` "
                "already applies. NOT a kernel parameter -- it is a property of the INPUT, and it "
                "earns an axis because the LayerNorm here consumes `x` DIRECTLY, so the caller "
                "controls the conditioning of every row it normalizes. MEASURED: the worst "
                "|err|/bound ratio rises 1.34x from 0 to 100, which is small next to standalone "
                "LayerNorm's ~850x but is not flat -- and flat is what a waiver would be claiming"
            ),
            values=(0.0, 10.0, 100.0),
            facets={
                # NOT "zero_mean": `_build` gives each row its own offset in [-2.4, 2.4] so a row
                # normalized with its neighbour's statistics is visible at all. This axis is the
                # GLOBAL offset on top of that, and 0.0 means "none", not "centred".
                "no_global_offset": lambda v: v == 0.0,
                "off_centre": lambda v: v != 0.0,
                "badly_conditioned": lambda v: abs(v) >= 100.0,
            },
        ),
    ),
    # Order MIRRORS the front door's check order: regions are matched first-wins, so a combo
    # violating two rules must report the one listed first here.
    # LayerNorm folded into the dual-gated projection: a row reduction, two contractions
    # and a sigmoid gate.
    computes=("row_reduction", "contraction", "saturating_activation"),
    unsupported=(
        Unsupported(
            where=lambda ab_dtype: ab_dtype == torch.float32,
            raises=ValueError,
            match=r"must be 16-bit",
            reason=(
                "the gated post-activation is stored through stmatrix, which has a 16-bit form "
                "only, and the SM90 WGMMA atom has no fp32 form either. fp32 nonetheless passes "
                "the dtype-membership check -- it is the LayerNorm gain's own dtype -- so without "
                "this guard it would be refused several frames into tracing"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, fusion_variant, pingpong: (
                ab_dtype != torch.float32 and pingpong and fusion_variant == "prolog_ln"
            ),
            raises=ValueError,
            match=r"pingpong=True is not supported",
            reason=(
                "the prologue's reduction spans every MMA warpgroup and publishes through the "
                "epilogue barrier, which ping-pong sizes for ONE -- 256 threads arriving at a "
                "128-thread barrier means the second half waits forever. A HANG, so it is refused "
                "at the FRONT DOOR: the functor's own assert would also stop it, but an assert "
                "cannot satisfy a region -- the point is to oblige a checkable refusal. The "
                "algebraic fold reduces warpgroup-locally and is the variant that accepts it"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, tile_M: ab_dtype != torch.float32 and tile_M == 320,
            raises=ValueError,
            match=r"tile_M must be one of",
            reason=(
                "the fused reduction tiles tile_M by its row-group count, and the base splits 320 "
                "across two warpgroups -- 128 groups, which 320 is not a multiple of. Before this "
                "region it was an AssertionError from inside stats_tiled_copy naming a layout, "
                "three frames below the call and with no mention of tile_M"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, chunk_g: (
                ab_dtype != torch.float32 and chunk_g != 1 and chunk_g % 16 != 0
            ),
            raises=ValueError,
            match=r"multiple of 16",
            reason=(
                "16 is the stmatrix atom's N-width. A chunk narrower than that puts the up/gate "
                "split INSIDE an atom, where the register half-split the epilogue relies on does "
                "not exist -- so the fold would pair registers from different columns and return a "
                "plausible wrong answer with no diagnostic"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, chunk_g, bias: (
                ab_dtype != torch.float32
                and (chunk_g == 1 or chunk_g % 16 == 0)
                and bias == "gate_only"
            ),
            raises=ValueError,
            match=r"both be given or both omitted",
            reason=(
                "a bias on one projection only biases half the pre-activation. Nothing downstream "
                "can detect it: the kernel would add the gate bias, leave the up projection "
                "unbiased, and gate the two together into a result that looks entirely reasonable"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, chunk_g, bias, gate3, fusion_variant, pingpong, x_gate: (
                x_gate
                and gate3
                and ab_dtype == torch.bfloat16
                and chunk_g == 1
                and bias == "both"
                and not pingpong
            ),
            raises=ValueError,
            match=r"mutually exclusive",
            reason=(
                "both ask this kernel to compute a SECOND output, and the two-A path has no "
                "accumulator left for an output gate. Refusing is the whole content: silently "
                "dropping one would produce a complete, plausible tensor with a term missing. "
                "Swept on BOTH fusions: the refusal is a property of the two-A shape, which both "
                "of them have, so pinning it to one would leave the other's front door untested"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, chunk_g, bias, gate3, fusion_variant, pingpong, x_gate: (
                x_gate
                and chunk_g == 16
                and ab_dtype == torch.bfloat16
                and bias == "both"
                and not gate3
                and not pingpong
            ),
            raises=ValueError,
            match=r"x_gate requires chunk_g=1",
            reason=(
                "the block interleave exists to pair an up column with a gate column inside ONE 2N "
                "accumulator; the two-A path has two N-wide accumulators and nothing to pair. "
                "chunk_g=8 is excluded because the multiple-of-16 region above claims it first. "
                "Swept on both fusions, for the reason the gate3 region above gives"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, chunk_g, bias, gate3, fusion_variant, pingpong, x_gate: (
                x_gate
                and pingpong
                and ab_dtype == torch.bfloat16
                and chunk_g == 1
                and bias == "both"
                and not gate3
                and fusion_variant == "alg_fold"
            ),
            raises=ValueError,
            match=r"x_gate requires pingpong=False",
            reason=(
                "a pipeline stage holds FOUR operand buffers here, so the stage count is already "
                "halved; two warpgroups alternating over them is not a schedule this was measured "
                "at. Declared rather than left to the kernel's own assert, which fires inside "
                "tracing. **Pinned to alg_fold, and unlike the two regions above that pin is "
                "NECESSARY**: with prolog_ln the pingpong-versus-variant region near the top claims "
                "the combo first and reports a different message, which mirrors the front door's "
                "own check order"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, chunk_g, bias, gate3, fusion_variant, arg_fault: (
                ab_dtype == torch.bfloat16
                and chunk_g == 1
                and bias == "both"
                and not gate3
                and fusion_variant == "prolog_ln"
                and arg_fault == "mask_dtype"
            ),
            raises=ValueError,
            match=r"unsupported dtype for mask",
            reason=(
                "a broadcast vector's dtype is only a LOAD WIDTH -- nothing about it constrains the "
                "MMA atom -- so nothing downstream refuses it, and before the front door checked it "
                "the only thing that failed was the dict lookup building the compile key. That is a "
                "bare KeyError naming a torch dtype and no argument"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, chunk_g, bias, gate3, fusion_variant, arg_fault: (
                ab_dtype == torch.bfloat16
                and chunk_g == 1
                and bias == "both"
                and not gate3
                and fusion_variant == "prolog_ln"
                and arg_fault == "mask_extent"
            ),
            raises=ValueError,
            match=r"mask must be",
            reason=(
                "the mask is broadcast along N with stride 0, so a mask whose length disagrees with "
                "M is read as if it were the right length: every row past the end takes whatever "
                "follows it in memory, and the output is plausible and wrong"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, chunk_g, bias, gate3, fusion_variant, arg_fault: (
                ab_dtype == torch.bfloat16
                and chunk_g == 1
                and bias == "both"
                and not gate3
                and fusion_variant == "prolog_ln"
                and arg_fault == "bias_extent"
            ),
            raises=ValueError,
            match=r"bg must be",
            reason=(
                "same failure on the other axis: a projection bias shorter than N biases some "
                "columns with a neighbour's value. Declared separately from the mask because the "
                "two are checked at different points and a single region would not tell them apart"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, chunk_g, bias, gate3, fusion_variant, arg_fault: (
                ab_dtype == torch.bfloat16
                and chunk_g == 1
                and bias == "both"
                and not gate3
                and fusion_variant == "prolog_ln"
                and arg_fault == "postact_extent"
            ),
            raises=ValueError,
            match=r"PostAct must be",
            reason=(
                "the destination is written through a TMA descriptor built from the extents the "
                "caller declared, so a short one is an out-of-bounds STORE rather than a wrong read "
                "-- the failure class that corrupts an unrelated allocation"
            ),
        ),
    ),
)

#: The A2A-fused TriMul input projection's own geometry, not an arbitrary small one. In that
#: workflow the activation is ``(tokens, D)`` and BOTH projections are ``(D, D)`` -- so ``K == N ==
#: D`` and the kernel's pre-activation is ``2D`` wide. The default cell holds that relation exactly:
#: ``K = N = D`` and ``tile_N = 2D``.
#:
#: ``D = 128`` is the smallest feature width the workflow runs, chosen because K is the one extent
#: this kernel BAKES (the staged gain is a shared-memory allocation), so every additional D is
#: another compile in every cell. The other three -- 256, 384, 512 -- are swept by
#: ``test_the_fused_projection_accepts_off_grid_and_odd_extents`` and pinned end to end against the
#: kernel this reproduces by the bit-identity sweep in ``benchmark/``.
_D = 128
_N, _K = _D, _D
#: The CTA tile over the 2N PRE-activation, so the output tile is half of it.
_TILE_M, _TILE_N = 128, 2 * _D
#: Token count. **Deliberately large enough to reach the MULTI-WAVE persistent regime**: with
#: ``2N == tile_N`` the grid is ``M / tile_M`` work tiles over ~132 resident CTAs, so ``M = 32768``
#: gives 256 tiles and every CTA sees at least two. That regime is not a stress test, it is where
#: this kernel's one genuine hazard lives -- a persistent CTA reuses ONE statistics scratch across
#: tiles, so tile n+1's finalize can overwrite mu/rstd while tile n's normalize is still reading
#: them. ``finalize_row_stats``'s leading barrier exists for exactly that, and at a token count
#: small enough to give each CTA a single tile the guard is never exercised at all.
#:
#: M is free of the compile key (it enters as a symbol), so raising it costs run time only -- about
#: 10 us of kernel per cell.
_M = 32768


def _pitch_pad(rows, cols, dtype, fill=0.0):
    """Allocate a ``(rows, cols)`` view whose row pitch meets the 16-byte TMA floor.

    Args:
        rows: Leading extent. Unconstrained.
        cols: Trailing extent. Unconstrained -- that is the point.
        dtype: Element type; the floor is ``16 // itemsize`` elements.
        fill: Value the whole buffer (padding included) starts at.

    Returns:
        A ``(rows, cols)`` VIEW of a padded buffer. The caller must keep it alive; it holds the
        only reference to the backing storage.
    """
    e = 16 // dtype.itemsize
    pad = (cols + e - 1) // e * e
    return torch.full((rows, pad), fill, device="cuda", dtype=dtype)[:, :cols]


def _build(
    M=_M,
    N=_N,
    K=_K,
    dtype=torch.bfloat16,
    ln_bias=True,
    bias="both",
    mask=False,
    N3=0,
    x_gate=False,
    row_mean=0.0,
    seed=0,
):
    """Build one cell's operands.

    Args:
        M: Token extent. Unconstrained -- off-grid values are the interesting ones.
        N: Output feature extent, i.e. HALF the pre-activation width. Unconstrained.
        K: Contraction extent, the LayerNorm axis. Must meet the 16-byte pitch floor.
        dtype: Operand and output element type.
        ln_bias: Whether to build a ``(K,)`` LayerNorm bias.
        bias: ``"none"``, ``"both"``, or ``"gate_only"`` (which the kernel must refuse).
        mask: Whether to build a per-row post-gate mask.
        N3: Output-gate width, or 0 for no output gate.
        x_gate: Whether to build the SEPARATE ``(M, K)`` gate activation. It is drawn INDEPENDENTLY
            of ``x`` rather than derived from it, and that is the point: a kernel that fed the gate
            ``LN(x)`` instead of ``x_gate`` would still produce a plausible tensor, and only an
            unrelated gate input makes the two distinguishable.
        row_mean: A constant added to every element of `x`, on top of the per-row offset above, so
            each row's mean moves together. In exact arithmetic the LayerNorm subtracts it back out
            and the output is unchanged; in floating point it is a cancellation whose conditioning
            is proportional to ``|mu|/sigma``, which is the regime a padded-tail defect lives in and
            the one ``torch.randn`` never reaches. Swept by the ``row_mean`` axis.
        seed: RNG seed, so a failure is reproducible.

    Returns:
        A dict of operands, keyed by the front door's parameter names. ``x`` is given a per-row
        offset on purpose: ``torch.randn`` has a near-zero row mean, which lets a dropped centring
        look almost correct.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=dtype, generator=g)
    # A per-row offset, so no two rows in a CTA tile share a mean -- a row normalized with its
    # neighbour's statistics has to be VISIBLE, and against zero-mean input it is not. The offset is
    # WRAPPED rather than proportional to the row index: an offset that grows with M reaches ~330 at
    # M=32768, where fp16's ulp is 0.25, so the row becomes near-constant, its variance a
    # catastrophic cancellation, and its rstd enormous. That is an ill-conditioned INPUT, not a
    # kernel defect, and it is outside what any of these bounds model (see
    # `fused_ln_gated_error_bound`). 97 is coprime with every tile height in the pool, so the
    # pattern never aligns with a tile boundary.
    x = x + (torch.arange(M, device="cuda", dtype=dtype).reshape(-1, 1) % 97 - 48) * 0.05
    # A GLOBAL offset on top of the per-row one, swept by the `row_mean` axis. Applied after the
    # draw so the LayerNorm's own statistics see it; a shift the reference and the kernel both
    # subtract back out in exact arithmetic, and therefore a pure test of how each handles the
    # cancellation.
    if row_mean:
        x = x + row_mean
    scale = 1.0 / math.sqrt(K)
    out = {
        "x": x,
        "norm_weight": torch.randn(K, device="cuda", dtype=torch.float32, generator=g),
        "norm_bias": (
            torch.randn(K, device="cuda", dtype=torch.float32, generator=g) if ln_bias else None
        ),
        "Wg": torch.randn(N, K, device="cuda", dtype=dtype, generator=g) * scale,
        "Wp": torch.randn(N, K, device="cuda", dtype=dtype, generator=g) * scale,
        "bg": None,
        "bp": None,
        "mask": (torch.rand(M, device="cuda", dtype=dtype, generator=g) if mask else None),
        "W3": None,
        "b3": None,
        # ALREADY normalized by contract, so it is not put through a LayerNorm here. It carries no
        # per-row offset either: the offset above exists to make a mis-centred VALUE row visible,
        # and the gate side is never centred by this kernel at all.
        "x_gate": (torch.randn(M, K, device="cuda", dtype=dtype, generator=g) if x_gate else None),
    }
    if bias in ("both", "gate_only"):
        out["bg"] = torch.randn(N, device="cuda", dtype=dtype, generator=g)
    if bias == "both":
        out["bp"] = torch.randn(N, device="cuda", dtype=dtype, generator=g)
    if N3:
        out["W3"] = torch.randn(N3, K, device="cuda", dtype=dtype, generator=g) * scale
        out["b3"] = torch.randn(N3, device="cuda", dtype=torch.float32, generator=g)
    return out


def _apply_arg_fault(ops, fault):
    """Malform ONE tensor argument, as the ``arg_fault`` axis names it.

    Purpose
        Turns each declared fault into an actual call, so the region is swept by ``pytest.raises``
        rather than asserted by hand. Built here rather than in :func:`_build` because a fault is
        not a cell configuration -- every other axis describes a kernel the caller MEANT to ask for.

    Args:
        ops: The dict from :func:`_build`, copied rather than mutated so a caller may build once.
        fault: An ``arg_fault`` axis value. ``"none"`` returns the dict unchanged.

    Returns:
        A new dict with exactly one argument malformed.

    Raises:
        AssertionError: On an unknown fault name -- a value added to the axis without a way to
            build it would otherwise sweep as if it were covered.
    """
    if fault == "none":
        return ops
    out = dict(ops)
    M, N = out["x"].shape[0], out["Wg"].shape[0]
    if fault == "mask_dtype":
        out["mask"] = torch.rand(M, device="cuda", dtype=torch.float64)
    elif fault == "mask_extent":
        out["mask"] = torch.rand(M + 1, device="cuda", dtype=out["x"].dtype)
    elif fault == "bias_extent":
        if out["bp"] is None:
            return ops  # gate_only: the PAIRING error is the one declared for that cell
        out["bg"] = torch.randn(N + 1, device="cuda", dtype=out["x"].dtype)
        out["bp"] = torch.randn(N, device="cuda", dtype=out["x"].dtype)
    elif fault == "postact_extent":
        out["_postact_extent_fault"] = True
    else:
        raise AssertionError(f"unknown arg_fault {fault!r}: add a way to build it")
    return out


#: Sentinel for :func:`_run`'s ``tile_N``, because the natural tile depends on the path and a
#: literal default cannot express that: the dual tiles the 2N PRE-activation, the two-A ``x_gate``
#: path tiles the output width N directly. A sentinel keeps every existing caller byte-identical --
#: they resolve to ``_TILE_N`` exactly as before -- while an x_gate cell gets a tile that divides
#: what it actually tiles.
_TILE_N_NATURAL = object()


def _run(ops, *, tile_M=_TILE_M, tile_N=_TILE_N_NATURAL, transpose_out=False, **kw):
    """Allocate the outputs and run the front door on one cell.

    Args:
        ops: The dict from :func:`_build`. Its ``x_gate`` entry, when not None, selects the two-A
            path and is forwarded; the caller does not pass it separately.
        tile_M: CTA tile M.
        tile_N: CTA tile N. Over the 2N pre-activation on the dual path, over the output width N on
            the ``x_gate`` path. Left unset it resolves to ``_TILE_N`` for the dual (unchanged) and
            to this cell's own N for ``x_gate``.
        transpose_out: Allocate an m-major output view instead of an n-major one.
        **kw: Forwarded to the front door (``chunk_g``, ``blk_k``, ``fusion_variant``, ...).

    Returns:
        The dual output, or ``(dual, gate3)`` when the cell has an output gate. The dual output is
        always returned in its LOGICAL ``(M, N)`` orientation, so a caller can compare a transposed
        run against an untransposed one directly.
    """
    M, N = ops["x"].shape[0], ops["Wg"].shape[0]
    if tile_N is _TILE_N_NATURAL:
        tile_N = N if ops.get("x_gate") is not None else _TILE_N
    dt = ops["x"].dtype
    dual = _pitch_pad(N, M, dt).T if transpose_out else _pitch_pad(M, N, dt)
    if ops.get("_postact_extent_fault"):
        dual = _pitch_pad(M, N + 1, dt)  # one column too wide: an out-of-bounds STORE

    gate3_out = _pitch_pad(M, ops["W3"].shape[0], dt) if ops["W3"] is not None else None
    layernorm_dual_gated_gemm(
        ops["x"],
        ops["norm_weight"],
        ops["Wg"],
        ops["Wp"],
        dual,
        tile_M,
        tile_N,
        norm_bias=ops["norm_bias"],
        bg=ops["bg"],
        bp=ops["bp"],
        mask=ops["mask"],
        W3=ops["W3"],
        b3=ops["b3"],
        PostAct3=gate3_out,
        x_gate=ops.get("x_gate"),
        **kw,
    )
    torch.cuda.synchronize()
    return (dual, gate3_out) if gate3_out is not None else dual


def _prolog_ln_xgate_error_bound(
    x, x_gate, norm_weight, norm_bias, Wg, Wp, reference, eps, out_dtype, bg=None, bp=None
):
    """Per-element bound for the PHYSICAL two-A path: ``sigmoid(x_gate@Wg^T + bg) * (LN(x)@Wp^T + bp)``.

    Purpose
        The fourth bound this kernel family needs, and none of the three in
        `fold_cp_ops.testing.numerics` describes it. `alg_fold_xgate_error_bound` has the right
        SHAPE -- one arm folded, one arm plain -- but models the value arm as folded-and-repaired,
        which this variant does not do; `fused_ln_gated_error_bound` models the right value arm but
        gates it with a SECOND projection of the same ``LN(x)``, where here the gate reads an
        unrelated pre-normalized matrix. Either one is unsound in the direction that matters: it
        hands an arm an allowance for errors it cannot incur, so an arm that is genuinely wrong can
        sit inside it.

    Semantics
        Composed from the two models the kernel actually uses, one per arm:

        * **The gate arm is a PLAIN 16-bit GEMM.** ``x_gate`` arrives normalized, so there is no
          gain anywhere in its path, no statistic and no repair. Its error is the accumulation error
          alone -- identical to `alg_fold_xgate_error_bound`'s gate half, because the gate arm is
          the same kernel in both variants.
        * **The value arm is the physical fusion**, i.e. `fused_ln_gated_error_bound`'s treatment:
          the kernel does not multiply the exact ``LN(x)``, it normalizes in fp32 from fp32
          statistics and writes the result back at the ACTIVATION's width before the MMA. That
          perturbation is a rounding of an INPUT, so it reaches the output amplified by the weights
          and is carried as a widened pre-activation error rather than as an output tolerance.

        The two then meet as in every gated bound here: the gate's error through the sigmoid's
        maximum slope plus the hardware sigmoid's absolute error, scaled by the value magnitude;
        the value's error directly; one half-ulp of the store on top.

        **The ill-conditioned input regime `fused_ln_gated_error_bound` documents applies here
        too** -- a nearly-constant row makes ``E[x^2] - mu^2`` cancel catastrophically and this does
        not bound it. The harness keeps per-row means bounded, which is what makes that moot.

    Args:
        x: ``(M, K)`` un-normalized VALUE activation, in the kernel's operand dtype. Its dtype
            decides the width the perturbed operand is modelled at, so an fp32 copy silently
            tightens the bound past what the kernel can meet.
        x_gate: ``(M, K)`` PRE-NORMALIZED gate activation, same shape and dtype as `x`. Must be the
            tensor the kernel was given, not ``LN(x)`` recomputed: it is read raw, so its own
            rounding is already in it. Passing `x` for both -- the shape-compatible mistake --
            bounds a kernel this one is not.
        norm_weight: ``(K,)`` fp32 LayerNorm gain. Applies to the value arm only.
        norm_bias: ``(K,)`` fp32 LayerNorm bias, or None. Must match what the kernel was given: its
            absence removes a term rather than zeroing one.
        Wg: ``(N, K)`` gate weight, RAW. Same dtype as `x`.
        Wp: ``(N, K)`` value weight, RAW -- this variant folds nothing into it.
        reference: The exact fp64 ``(M, N)`` reference for the COMBINED output, biases and any
            post-gate mask already applied, or the store term is taken against the wrong magnitude.
        eps: The LayerNorm variance floor the kernel was given, or the perturbed operand is modelled
            against a different normalization than the one that ran.
        out_dtype: The dtype the combined output is stored as.
        bg: ``(N,)`` gate projection bias, or None. Sizes the gate magnitude only.
        bp: ``(N,)`` value projection bias, or None. This one matters more -- the value magnitude
            multiplies the gate's error.

    Returns:
        An fp64 tensor shaped like `reference`: the maximum absolute deviation each element may
        show.

    Raises:
        ValueError: If ``K * u >= 1``, where the gamma formula has no meaning, or if `Wg`/`Wp` or
            `x`/`x_gate` disagree in shape or dtype.

    Note:
        It reads four private constants out of `fold_cp_ops.testing.numerics` rather than
        re-deriving them. That is deliberate: they are the SAME hardware facts its three sibling
        bounds use, and a second copy that drifted would make two bounds disagree about the
        sigmoid's slope. This function belongs beside those siblings and is here only because this
        change does not own that file.
    """
    K = x.shape[-1]
    if K * _U_FP32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    if Wg.shape != Wp.shape or Wg.dtype != Wp.dtype:
        raise ValueError(
            f"Wg{tuple(Wg.shape)}/{Wg.dtype} and Wp{tuple(Wp.shape)}/{Wp.dtype} must match: both "
            "are (N, K) operands of the same kernel at one width"
        )
    if x.shape != x_gate.shape or x.dtype != x_gate.dtype:
        raise ValueError(
            f"x{tuple(x.shape)}/{x.dtype} and x_gate{tuple(x_gate.shape)}/{x_gate.dtype} must "
            "match: the two-A kernel contracts both against the same K at one width"
        )
    gamma = (K * _U_FP32) / (1.0 - K * _U_FP32)
    # Gate arm: a plain 16-bit GEMM of an already-normalized activation. No gain, no repair.
    e_gate = gamma * (x_gate.double().abs() @ Wg.double().abs().transpose(-1, -2))
    # Value arm: the physical fusion's perturbed operand, exactly as fused_ln_gated_error_bound
    # models it -- normalized in fp32, stored back at the activation's width before the MMA.
    nb64 = None if norm_bias is None else norm_bias.double()
    an = torch.nn.functional.layer_norm(x.double(), (K,), norm_weight.double(), nb64, eps)
    ahat = torch.nn.functional.layer_norm(
        x.float(),
        (K,),
        norm_weight.float(),
        None if norm_bias is None else norm_bias.float(),
        eps,
    ).to(x.dtype)
    rel_stat = gamma + 2.0**-22 + 4.0 * _U_FP32
    dA = (ahat.double() - an).abs() + rel_stat * an.abs()
    wp64 = Wp.double().abs().transpose(-1, -2)
    e_up = gamma * (ahat.double().abs() @ wp64) + (dA @ wp64)
    abs_up = (ahat.double().abs() + dA) @ wp64
    if bp is not None:
        abs_up = abs_up + bp.double().abs()
    gate_term = (_SIGMOID_MAX_SLOPE * e_gate + _SIGMOID_ABS_ERR) * (abs_up + e_up)
    pre_store = gate_term + e_up
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    return pre_store + half_ulp * (reference.double().abs() + pre_store)


def _assert_close(got, ops, what="dual", fusion_variant="prolog_ln"):
    """Compare the DUAL output against its per-element bound, element by element.

    Purpose
        The bound is a TENSOR, not a scalar, and that is the point: an output element whose products
        cancel to near zero gets a correspondingly tight allowance, where a
        max-error-over-max-reference ratio would hand it the whole tensor's slack. That element is
        the one a fused kernel is most likely to get wrong.

    Semantics
        `fused_ln_gated_error_bound` carries the fusion's OWN extra error -- the kernel writes the
        normalized activation back at 16 bits before the MMA, and that is a rounding of an INPUT, so
        it reaches the output amplified by the weights rather than as an output tolerance. Every
        other term (the two accumulations, the hardware sigmoid's ``2**-12``, the store's half-ulp)
        is the shared gated derivation.

    Args:
        got: The kernel output, logically ``(M, N)``.
        ops: The dict from :func:`_build`, for the reference and the bound.
        what: Label for the failure message.
        fusion_variant: Which fusion produced `got`, and therefore which bound applies. The two
            reach the same result by different arithmetic, so the bounds are NOT interchangeable in
            either direction -- `alg_fold_error_bound` models a rounding of the WEIGHT that
            `prolog_ln` does not perform, and `fused_ln_gated_error_bound` models a rounding of the
            normalized ACTIVATION that `alg_fold` does not. Passing the wrong one is unsound rather
            than merely loose: it can pass a kernel that is wrong and fail one that is right.

    Returns:
        The worst observed ``|err| / bound`` ratio, so a caller may print how close the cell runs.

    Raises:
        AssertionError: From `assert_elementwise`, naming the violating count, the fraction and the
            worst three offenders with their coordinates.
    """
    ref = layernorm_dual_gated_gemm_ref(
        ops["x"],
        ops["norm_weight"],
        ops["Wg"],
        ops["Wp"],
        norm_bias=ops["norm_bias"],
        bg=ops["bg"],
        bp=ops["bp"],
        mask=ops["mask"],
        x_gate=ops.get("x_gate"),
    ).double()
    common = (ops["x"], ops["norm_weight"], ops["norm_bias"], ops["Wg"], ops["Wp"], ref, 1e-5)
    if ops.get("x_gate") is not None:
        # A bound of its OWN per variant, not a variant of the one-activation ones: this kernel's
        # two arms are produced by different arithmetic -- the gate is a plain 16-bit GEMM of an
        # already-normalized activation, the value is whichever fusion ran -- and a bound that
        # modelled both arms alike is unsound in the pass-a-wrong-kernel direction on the gate arm.
        # The two two-A bounds share the gate half exactly and differ only in the value half, which
        # is the same split the kernels themselves have.
        xgate_bound = (
            alg_fold_xgate_error_bound
            if fusion_variant == "alg_fold"
            else _prolog_ln_xgate_error_bound
        )
        bound = xgate_bound(
            ops["x"],
            ops["x_gate"],
            ops["norm_weight"],
            ops["norm_bias"],
            ops["Wg"],
            ops["Wp"],
            ref,
            1e-5,
            ops["x"].dtype,
            bg=ops["bg"],
            bp=ops["bp"],
        )
    elif fusion_variant == "alg_fold":
        # The projection biases enter this bound and not the other one: the fold emits them
        # interleaved alongside `c` and `d`, so they are added at a different point in the
        # arithmetic and carry their own rounding.
        bound = alg_fold_error_bound(*common, ops["x"].dtype, bg=ops["bg"], bp=ops["bp"])
    else:
        bound = fused_ln_gated_error_bound(*common, ops["x"].dtype)
    if ops["mask"] is not None:
        # The mask multiplies the fp32 output AFTER the gate, so it scales the error that reached
        # that point. Scaling the BOUND by it (rather than leaving the unmasked one) is what keeps a
        # masked-to-near-zero row held to a tight allowance instead of the unmasked row's.
        bound = bound * ops["mask"].double().abs().reshape(-1, 1)
        # Half an ulp of the RE-STORED value. The relative form alone is wrong near zero: below the
        # smallest normal the exponent stops shrinking and the spacing becomes ABSOLUTE, so a
        # subnormal output's half-ulp is `tiny * 2**-bits`, not `|ref| * 2**-bits`. A row whose mask
        # is near zero lands exactly there -- the mask has already crushed every other term -- and
        # the relative form then holds a correctly-rounded subnormal to a fraction of the only gap
        # the format has. Measured: fp16 outputs at 5 subnormal ulps failed a bound 4x tighter than
        # one ulp. `maximum`, not a sum, because half an ulp IS the larger of the two, not both.
        half_ulp = 2.0 ** -_PRECISION_BITS_OUT[ops["x"].dtype]
        subnormal_half_ulp = torch.finfo(ops["x"].dtype).tiny * half_ulp
        bound = bound + torch.clamp(half_ulp * ref.abs(), min=subnormal_half_ulp)
    return assert_elementwise(got, ref, bound, what=what)


def _assert_close_gate3(got3, ops, what="gate3", fusion_variant="prolog_ln"):
    """Compare the OUTPUT GATE against its own per-element bound.

    Its error reaches the output only through the sigmoid's slope, so the allowance is far tighter
    than the dual's -- and it has to be: the output lives in ``(0, 1)``, where a loose bound would
    accept nearly anything.

    Args:
        got3: The gate output, ``(M, N3)``.
        ops: The dict from :func:`_build`.
        what: Label for the failure message.
        fusion_variant: Which fusion produced `got3`, selecting its bound. The gate reads the same
            ``LN(x)`` as the dual, so it inherits the same fusion-specific error sources and the
            same rule: the two bounds are not interchangeable in either direction.

    Returns:
        The worst observed ratio.

    Raises:
        AssertionError: From `assert_elementwise`.
    """
    _, ref3 = layernorm_dual_gated_gemm_ref(
        ops["x"],
        ops["norm_weight"],
        ops["Wg"],
        ops["Wp"],
        norm_bias=ops["norm_bias"],
        bg=ops["bg"],
        bp=ops["bp"],
        W3=ops["W3"],
        b3=ops["b3"],
    )
    ref3 = ref3.double()
    common = (ops["x"], ops["norm_weight"], ops["norm_bias"], ops["W3"], ref3, 1e-5)
    if fusion_variant == "alg_fold":
        bound = alg_fold_sigmoid_error_bound(*common, ops["x"].dtype, b3=ops["b3"])
    else:
        bound = fused_ln_sigmoid_error_bound(*common, ops["x"].dtype)
    return assert_elementwise(got3, ref3, bound, what=what)


#: Mantissa bits of each supported output dtype, for the one place this file needs a half-ulp of its
#: own: re-rounding after the post-gate mask. Mirrors ``numerics._PRECISION_BITS`` rather than
#: importing a private name.
_PRECISION_BITS_OUT = {torch.bfloat16: 8, torch.float16: 11}


# ------------------------------------------------------------------ reference gates


@matrix_exempt("a pure function of k; no kernel, no shape axis to sweep")
def test_the_k_tiling_pads_to_64_before_choosing_the_tile():
    """Every ``k`` lands on the upstream's tiling: ``BLK_K=64`` over ``ceil(k/64)*64``.

    This is an ARITHMETIC assertion wearing a tiling costume. The tile decides how the fp32
    accumulation is grouped, so two tilings of one contraction differ in the last bits -- and this
    project is a reorganization, whose contract is byte-identity with the upstream.

    It is pinned because it was WRONG, and wrong in a way that looked defensive: `auto_blk_k`
    refuses a ``k`` no allowed tile divides, and the caller caught that refusal with a fallback to
    the narrowest tile. Every ``k`` that is not a multiple of 64 was silently re-tiled -- at
    ``k=136``, nine 16-wide k-tiles against the upstream's three 64-wide ones -- and it cost one
    element in 65536 by one bf16 ulp. The K pool held only multiples of 64, so nothing ran it.

    A pinned ``blk_k`` is honoured as given: an explicit tile is a deliberate choice about
    accumulation grouping, and overriding it would make the parameter a suggestion.
    """
    for k in (128, 192, 256, 384, 512):  # already a multiple of 64: nothing to pad
        assert resolve_k_tiling(k) == (64, k), f"k={k} must be unchanged"
    for k, gemm_k in ((136, 192), (144, 192), (176, 192), (200, 256), (264, 320)):
        assert resolve_k_tiling(k) == (64, gemm_k), (
            f"k={k} must pad to {gemm_k} and tile with 64, as the upstream does -- not fall back "
            f"to a narrow tile because no tile divides k itself"
        )
    assert resolve_k_tiling(136, blk_k=16) == (16, 144), "a pinned tile must be honoured as given"
    for k in (128, 136, 200, 264):
        blk_k, gemm_k = resolve_k_tiling(k)
        assert gemm_k % blk_k == 0, f"k={k}: the tile must divide the padded extent exactly"
        assert gemm_k >= k, f"k={k}: padding must never truncate the contraction"


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "ab_dtype",
    "chunk_g",
    "ln_bias",
    "bias",
    "mask",
    drop={
        "ab_dtype": (torch.float32,),
        "chunk_g": (8,),
        "bias": ("gate_only",),
    },
    because=(
        "the dropped values are exactly the declared unsupported regions, which "
        "test_unsupported_combos_raise sweeps instead -- they have no correct output to compare "
        "against"
    ),
)
def test_the_fused_projection_matches_the_reference(ab_dtype, chunk_g, ln_bias, bias, mask):
    """The fused output matches an unfused fp32 LayerNorm-then-GEMM within the derived bound."""
    ops = _build(dtype=ab_dtype, ln_bias=ln_bias, bias=bias, mask=mask)
    _assert_close(_run(ops, chunk_g=chunk_g), ops)


@requires_sm90
@LN_DUAL_GATED.parametrize("row_mean", "fusion_variant")
def test_off_centre_rows_stay_inside_the_bound(row_mean, fusion_variant):
    """Shifting every row's mean must not degrade agreement, and BOTH fusions are checked.

    The LayerNorm here consumes ``x`` directly, so the caller sets the conditioning of every row it
    normalizes -- which makes the row mean a property of the INPUT that the shape axes cannot vary.
    Two things would be invisible without it: a padded tile whose masked lanes contribute something
    rather than nothing (an error proportional to ``mean**2``, hence exactly zero at ``torch.randn``
    's mean), and the algebraic fold's own cancellation, which subtracts two large nearly-equal
    quantities and is weakest precisely where ``|mu|/sigma`` is largest.

    MEASURED before this test existed: the worst ratio rises from 0.239 to 0.321 between ``mu=0``
    and ``mu=100`` -- a 1.34x, small next to standalone LayerNorm's ~850x but not flat, which is
    what a waiver would have been claiming. ``mu=1000`` is deliberately NOT in the pool: at bf16 the
    spacing at 1000 is ~3.9, so a unit-variance row quantizes to a handful of levels and its
    variance becomes quantization noise. That is an ill-conditioned INPUT, not a kernel defect.
    """
    ops = _build(row_mean=row_mean)
    got = _run(ops, fusion_variant=fusion_variant)
    _assert_close(got, ops, what=f"row_mean={row_mean:g}", fusion_variant=fusion_variant)


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "ab_dtype",
    "chunk_g",
    "ln_bias",
    "bias",
    "mask",
    "pingpong",
    drop={"ab_dtype": (torch.float32,), "chunk_g": (8,), "bias": ("gate_only",)},
    because=(
        "the dropped values are exactly the declared unsupported regions, which "
        "test_unsupported_combos_raise sweeps instead"
    ),
)
def test_the_algebraic_fold_matches_the_reference(ab_dtype, chunk_g, ln_bias, bias, mask, pingpong):
    """The ALGEBRAIC fusion matches the same reference, against its OWN bound.

    The `prolog_ln` arm above cannot stand in for this. The two fusions share a front door, a
    reference and a scheduler, but not one line of the arithmetic that produces a number: this one
    never forms ``LN(x)`` at all, multiplying the raw activation by a pre-folded weight and
    repairing the difference rank-one in the epilogue. So every term of the error is different, and
    so is every way it can be wrong.

    **Crossing `ln_bias` is the load-bearing part**, not a completeness gesture. The LayerNorm bias
    is the term the fold turns into a SEPARATE epilogue operand ``d = b_ln @ B``, which exists only
    when the bias does -- and its absence has to be a compile key, or the artifact is traced
    demanding a tensor the front door passes as None. That mismatch is a hard `TypeError` naming
    only an epilogue-argument index, and nothing above this line would have produced it: before this
    test, no correctness cell in this module ran `alg_fold` at any setting.
    """
    ops = _build(dtype=ab_dtype, ln_bias=ln_bias, bias=bias, mask=mask)
    # Ping-pong has a NARROWER tile-N ceiling (208) than the cooperative schedule, because two MMA
    # warpgroups alternate rather than share one tile. The module default is `2 * _D` = 256, which
    # the front door correctly REFUSES at pingpong=True -- so the ping-pong arm is run at the
    # widest tile it accepts, not at a geometry it is entitled to reject.
    tile_N = 128 if pingpong else _TILE_N_NATURAL
    got = _run(ops, chunk_g=chunk_g, fusion_variant="alg_fold", pingpong=pingpong, tile_N=tile_N)
    _assert_close(got, ops, fusion_variant="alg_fold", what=f"pingpong={pingpong}")


@requires_sm90
@LN_DUAL_GATED.parametrize("ln_bias", "transpose_out")
def test_the_algebraic_folds_output_gate_is_corrected_like_the_dual(ln_bias, transpose_out):
    """The OUTPUT GATE reads the same ``LN(x)``, so it needs the same rank-one repair.

    Under this fusion the gate's accumulator is as raw as the dual's, and the base's gate arm --
    written for a kernel that normalized its activation in the prologue -- adds ``b3`` and applies
    the sigmoid to whatever it is handed. Inheriting it unchanged produced a **fully-formed,
    finite, entirely wrong** gate beside a correct dual output: no fault, no NaN, and every
    structural test still green.

    **Both outputs are checked, and the pairing is the point.** The dual came out right through the
    whole defect, so a test that looked only at the dual -- or only at "does it run" -- would have
    passed. `transpose_out` is crossed in because it moves the DUAL's store to an m-major layout
    while the gate's stays row-major, which is the one geometry where the two regions disagree
    about their output tensor.
    """
    ops = _build(N3=_N, ln_bias=ln_bias, bias="both")
    dual, gate3 = _run(ops, transpose_out=transpose_out, fusion_variant="alg_fold")
    _assert_close(dual, ops, fusion_variant="alg_fold")
    _assert_close_gate3(gate3, ops, fusion_variant="alg_fold")


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "tile_M",
    "fusion_variant",
    "mask",
    drop={"tile_M": (320,)},
    because="320 is the declared unsupported region, which test_unsupported_combos_raise sweeps",
)
def test_every_cta_tile_m_is_correct_including_the_multi_subtile_ones(tile_M, fusion_variant, mask):
    """Every accepted ``tile_M``, on both fusions. The axis exists because 256 was WRONG.

    The number of epilogue subtiles along M is a function of ``tile_M``, and
    `SmemColVecBroadcast.begin_loop` used to return its whole-tile fragment without slicing by the
    subtile coordinate. That is correct at 64 and 128 -- one M subtile -- and at 256 it handed
    every subtile the FIRST one's rows, so most of the output got another row's ``mu``/``rstd``.
    Finite, plausible, and wrong by ~20%, from a legal front-door argument.

    **Only `alg_fold` could see it**, which is why running one fusion is not enough: `prolog_ln`
    reads the same statistics in the MAINLOOP through the partition that wrote them, so a permuted
    read cancels; the fold reads them in the EPILOGUE, indexed by the true tile row.

    ``mask`` is crossed because it is the other column-vector on this path -- a GMEM one, which
    slices correctly and was the control that localized the bug to the SMEM op.
    """
    # tile_N is pinned at 128 rather than the module default: at 192 the base splits the tile
    # across three warpgroups only while tile_N <= 128, and past that the row-group count stops
    # dividing tile_M. That interaction is its own declared refusal, not this test's subject.
    ops = _build(bias="both", mask=mask)
    _assert_close(
        _run(ops, tile_M=tile_M, tile_N=128, fusion_variant=fusion_variant),
        ops,
        fusion_variant=fusion_variant,
    )


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "tile_M",
    "mask",
    drop={"tile_M": (256, 320)},
    because=(
        "ping-pong accepts tile_M in {64, 128, 192} only, which the base enforces; 256 and 320 are "
        "cooperative-only tiles and their refusal belongs to the base, not to this fusion"
    ),
)
def test_the_ping_pong_schedule_agrees_with_the_cooperative_one(tile_M, mask):
    """Two alternating warpgroups must produce the same answer as two sharing a tile.

    Ping-pong is a SCHEDULE, so the arithmetic should not move -- but three things do move with it,
    and each is a wrong answer rather than a slowdown if it moves wrongly. The row reduction becomes
    warpgroup-LOCAL, so a thread owns more rows; the statistics scratch becomes one block PER
    warpgroup, because the two are on different work tiles at once; and the publish barrier becomes
    warpgroup-private, because the epilogue barrier ping-pong sizes for one warpgroup would fit a
    warpgroup-local reduction EXACTLY and silently rendezvous the two with each other.

    Compared against the cooperative run rather than only against the reference, so a change that
    moved both by the same amount would still fail. Not `assert_bitwise`: the reduction genuinely
    sums a different number of rows per thread, so the summation ORDER differs and equality is the
    wrong claim -- both are held to the fold's own bound instead.

    ``tile_N`` is pinned at 128: ping-pong caps it well below the cooperative limit and the cap
    varies with ``tile_M`` (256 at 64, 208 at 128, 128 at 192), so 128 is the one value legal for
    every tile in the pool.
    """
    ops = _build(bias="both", mask=mask)
    coop = _run(ops, tile_M=tile_M, tile_N=128, fusion_variant="alg_fold")
    ping = _run(ops, tile_M=tile_M, tile_N=128, fusion_variant="alg_fold", pingpong=True)
    _assert_close(ping, ops, fusion_variant="alg_fold", what="pingpong")
    _assert_close(coop, ops, fusion_variant="alg_fold", what="cooperative")
    assert_elementwise(
        ping.double(),
        coop.double(),
        alg_fold_error_bound(
            ops["x"],
            ops["norm_weight"],
            ops["norm_bias"],
            ops["Wg"],
            ops["Wp"],
            coop.double(),
            1e-5,
            ops["x"].dtype,
            bg=ops["bg"],
            bp=ops["bp"],
        )
        * 2,
        what="pingpong vs cooperative",
    )


@requires_sm90
@LN_DUAL_GATED.parametrize("fusion_variant")
def test_the_output_gate_and_the_mask_coexist(fusion_variant):
    """A masked dual and an output gate in one launch, which the UPSTREAM refuses.

    `main`'s stagec entry asserts ``mask is None`` whenever ``W3`` is set. That constraint does not
    transfer: here the gate is an ordinary work tile of the same GEMM and the mask is a per-row
    multiplier the DUAL's epilogue applies after its gate, so the two never meet. Both fusions are
    checked, because "more capable than the thing we are reproducing" is a claim worth holding to
    the same bound as parity is.

    The gate is deliberately NOT masked: it multiplies a downstream result, so masking it here
    would apply the mask twice. That is what makes this a real combination rather than a
    coincidence of two disjoint features -- the two epilogue arms treat the same flag differently.
    """
    ops = _build(N3=_N, bias="both", mask=True)
    dual, gate3 = _run(ops, fusion_variant=fusion_variant)
    _assert_close(dual, ops, fusion_variant=fusion_variant)
    _assert_close_gate3(gate3, ops, fusion_variant=fusion_variant)


@requires_sm90
@LN_DUAL_GATED.parametrize("fusion_variant")
@pytest.mark.parametrize("tile_N", [96, 160], ids=["tile96", "tile160"])
def test_a_tile_that_does_not_divide_the_pre_activation_is_still_correct(fusion_variant, tile_N):
    """``tile_N`` need not divide ``2N`` -- the partial last tile is predicated, not forbidden.

    The upstream asserts ``2N % tile_N == 0`` unconditionally on its stagec entry. This kernel does
    not need that: only the OUTPUT GATE does, and for a different reason -- the gate's region must
    begin on a work-tile edge because its weight rows are appended to the dual's, which
    `test_an_output_gate_whose_region_boundary_is_off_tile_is_refused` covers.

    Worth a test rather than a shrug, because the missing assertion is indistinguishable from an
    oversight until someone runs it: an unpredicated partial tile would write past the output or
    fold a garbage column into the gate, and both are silent.
    """
    ops = _build(N=128, bias="both", mask=True)
    _assert_close(
        _run(ops, tile_N=tile_N, fusion_variant=fusion_variant), ops, fusion_variant=fusion_variant
    )


@requires_sm90
@LN_DUAL_GATED.parametrize("fusion_variant")
def test_a_transposed_store_hands_the_trimul_front_two_contiguous_halves(fusion_variant):
    """The ``[a|b]`` feed the upstream spells ``split_out_half``, which this API does not need.

    `main`'s stagec ALLOCATES and RETURNS its output, so producing two halves takes a flag that
    reshapes the return value. This front door takes ``PostAct`` as an out-parameter, so the caller
    already holds the buffer and the split is a slice of their own tensor -- the flag would return
    views of something they gave us.

    **But the property the flag exists for still has to hold**, and it is not free: with an m-major
    store the buffer is ``(2D, M)``, so ``buf[:D]`` and ``buf[D:]`` are each CONTIGUOUS ``(D, M)``
    and admit a zero-copy ``view`` into a downstream batched GEMM. That is the TriMul front
    emitting both operands in one invocation. Without the transpose they are strided views, still
    correct and still free, but not zero-copy -- which is why the workflow's front transposes.

    Asserted against two INDEPENDENT half-width launches, so the test would catch a store that put
    the halves in the wrong order as well as one that mixed them.
    """
    half = _N // 2
    ops = _build(bias="both")
    buf = _pitch_pad(_N, _M, ops["x"].dtype)
    layernorm_dual_gated_gemm(
        ops["x"],
        ops["norm_weight"],
        ops["Wg"],
        ops["Wp"],
        buf.T,
        _TILE_M,
        _TILE_N,
        norm_bias=ops["norm_bias"],
        bg=ops["bg"],
        bp=ops["bp"],
        eps=1e-5,
        fusion_variant=fusion_variant,
    )
    torch.cuda.synchronize()
    a, b = buf[:half], buf[half:]
    assert a.is_contiguous() and b.is_contiguous(), (
        "an m-major store must leave each half contiguous, or the front's zero-copy feed is gone"
    )
    assert a.view(1, half, _M).data_ptr() == a.data_ptr(), "the downstream view must be zero-copy"
    whole = _run(ops, fusion_variant=fusion_variant)
    assert_bitwise(a.T, whole[:, :half], what="[a|b] first half")
    assert_bitwise(b.T, whole[:, half:], what="[a|b] second half")


@requires_sm90
@LN_DUAL_GATED.parametrize("ln_bias", "fusion_variant")
def test_a_k_that_no_k_tile_divides_is_an_ordinary_shape(ln_bias, fusion_variant):
    """``K = 136`` meets the 16-byte floor and nothing else, and must run on both fusions.

    ``136`` is ``8 * 17``: 16-byte aligned for a 16-bit operand, so a legal K by the ONLY constraint
    this kernel is allowed to carry -- and divisible by no member of ``ALLOWED_BLK_K``. The front
    door pads the contraction to ``gemm_k = ceil(136/64)*64 = 192`` and tiles it with ``blk_k = 64``,
    leaving a partial final k-tile that the TMA zero-fills and `stage_ln_affine` zero-tails in
    shared memory. Both contribute 0 to the contraction AND to the row statistics, and the
    normalizer divides by the TRUE K, so the padding is exact rather than merely out of range.

    **Pad first, then pick the tile.** Asking for a tile that divides 136 finds none and falls to
    the narrowest, which tiles the mainloop nine ways where the upstream tiles it three: same answer
    to within a rounding, which is precisely the failure -- the fp32 accumulation groups differently
    and one element in 65536 came out one bf16 ulp from `main`. This project is a reorganization, so
    that is a defect, and it is the reason the order matters here.

    **That padded path had no test, and it was broken for BOTH fusions.** The front door
    zero-extended the LayerNorm gain to ``gemm_k`` before the call, but the compiled signature
    declares that argument at the TRUE ``K`` -- because `stage_ln_affine` reads only ``[0, k_real)``
    from global and writes the tail of its own SHARED copy -- so every K the padding actually fired
    on was refused with a shape error naming an argument index. Nothing in the pool reached it: the
    K axis carried five values and all five were multiples of 64.
    """
    ops = _build(M=4096, N=128, K=136, ln_bias=ln_bias, bias="both", mask=True)
    _assert_close(_run(ops, fusion_variant=fusion_variant), ops, fusion_variant=fusion_variant)


@requires_sm90
@LN_DUAL_GATED.parametrize("blk_k")
@pytest.mark.parametrize(
    "M,N,K",
    [(18769, 137, 128), (30011, 97, 192), (4096, 256, 256), (16384, 384, 384), (8192, 512, 512)],
    ids=["offgrid_137sq", "odd_all", "D256", "D384", "D512"],
)
def test_the_fused_projection_accepts_off_grid_and_odd_extents(blk_k, M, N, K):
    """Every extent is free; only the 16-byte PITCH floor and ``blk_k | K`` constrain anything.

    The cells are the workflow's, stressed. ``D256``/``D384``/``D512`` are the three TriMul feature
    widths the default cell does not carry (K is a compile key, so they cost a compile each and are
    swept here rather than in the matrix) with ``K == N == D`` held. ``offgrid_137sq`` is
    ``M = 137**2``, a real token count for an odd sequence length, against ``N = 137`` -- neither a
    power of two nor a tile multiple nor a multiple of 8. ``odd_all`` makes every extent hostile at
    once. The outputs are pitch-padded, which is the documented way to keep the extent arbitrary.

    The unfused sibling's equivalent test sweeps tile_N; this sweeps ``blk_k`` instead, because the
    K tile is the axis this kernel ADDS -- and it is the one that has to divide something, so a
    shape it cannot tile is the failure worth reaching for.
    """
    if K % blk_k:
        pytest.skip(f"blk_k={blk_k} does not divide K={K}; the front door refuses it by design")
    ops = _build(M, N, K, bias="both", mask=True)
    _assert_close(_run(ops, chunk_g=1, blk_k=blk_k), ops)


@requires_sm90
@LN_DUAL_GATED.parametrize("blk_k")
def test_every_k_tile_width_is_correct_even_though_they_differ_in_bits(blk_k):
    """All three tiles are correct. They are NOT expected to agree bitwise -- see the module note."""
    ops = _build(K=128, bias="both", mask=True)
    _assert_close(_run(ops, chunk_g=1, blk_k=blk_k), ops)


# ------------------------------------------------------------------ bitwise transport gates


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "ln_bias",
    "bias",
    "mask",
    drop={"bias": ("gate_only",)},
    because="the half-biased cell has no correct output to be identical to",
)
def test_the_two_weight_layouts_are_bitwise_identical(ln_bias, bias, mask):
    """``chunk_g`` 1 and 16 pair different registers to compute the same product -- bit for bit.

    The strongest gate in this file. The two layouts share the LayerNorm prologue, the mainloop and
    the accumulation order, and differ only in which two accumulator registers the gate combines. No
    reference and no tolerance are involved, so a mis-pairing cannot hide inside a bound.
    """
    ops = _build(ln_bias=ln_bias, bias=bias, mask=mask)
    assert_bitwise(_run(ops, chunk_g=16), _run(ops, chunk_g=1), what="chunk_g=16 vs chunk_g=1")


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "chunk_g",
    only={"chunk_g": (1, 16)},
    because="8 is the declared unsupported value and has no correct output",
)
def test_the_transposed_store_writes_the_same_values(chunk_g):
    """m-major and n-major stores differ in descriptor and atom, never in arithmetic."""
    ops = _build(bias="both", mask=True)
    n_major = _run(ops, chunk_g=chunk_g)
    m_major = _run(ops, chunk_g=chunk_g, transpose_out=True)
    assert m_major.stride() == (1, _M), "the transposed output should be m-major"
    assert_bitwise(m_major, n_major, what="m-major vs n-major store")


@requires_sm90
@LN_DUAL_GATED.parametrize("ln_bias")
def test_omitting_the_layernorm_bias_equals_passing_a_zero_one(ln_bias):
    """A pruned add and an add of zero must agree bit for bit -- in both directions of the sweep.

    The values must match because adding zero changes nothing; what differs is that one kernel
    compiled the add out entirely. Running it in both directions is what makes this a gate on the
    PRUNING rather than on the zero: when ``ln_bias`` is True the two runs are the same kernel and
    the assertion is trivially satisfied, which is exactly the control case.
    """
    ops = _build(ln_bias=ln_bias, bias="both")
    zeroed = dict(ops)
    zeroed["norm_bias"] = (
        ops["norm_bias"]
        if ops["norm_bias"] is not None
        else torch.zeros(ops["x"].shape[-1], device="cuda", dtype=torch.float32)
    )
    assert_bitwise(_run(zeroed, chunk_g=1), _run(ops, chunk_g=1), what="zero bias vs no bias")


@requires_sm90
@matrix_exempt(
    "compares a masked run against an unmasked one at a fixed cell: the property is about WHERE "
    "the mask is applied, which no axis value varies"
)
def test_the_mask_multiplies_after_the_gate_not_before():
    """Masking the INPUT and masking the OUTPUT differ, because the gate is nonlinear.

    A kernel that folded the mask into the pre-activation would pass a loose reference comparison on
    a mask of ones and fail here, where the mask is genuinely varied.
    """
    ops = _build(bias="both", mask=True)
    got = _run(ops, chunk_g=1)
    unmasked = dict(ops)
    unmasked["mask"] = None
    plain = _run(unmasked, chunk_g=1)
    expected = plain.float() * ops["mask"].float().reshape(-1, 1)
    assert_elementwise(
        got.float(),
        expected,
        0.05 * expected.abs().max().clamp(min=1.0),
        what="masked output against plain-times-mask",
    )


# ------------------------------------------------------------------ the fused output gate


@requires_sm90
@LN_DUAL_GATED.parametrize("ln_bias", "mask")
def test_the_output_gate_does_not_perturb_the_dual_output_it_shares_a_mainloop_with(ln_bias, mask):
    """Adding the output gate must leave the dual result BIT-IDENTICAL.

    This is what makes the region-aware design's central claim testable: an output-gate work tile is
    an ordinary work tile of a WIDER operand, so the dual tiles must see exactly what they saw
    before. A design that instead squeezed the gate into the dual tiles' epilogue, or that changed
    the pipeline depth to make room, would perturb them -- slightly, plausibly, and invisibly to a
    tolerance test.
    """
    ops = _build(ln_bias=ln_bias, bias="both", mask=mask, N3=_K)
    dual_gated, _ = _run(ops, chunk_g=1)
    plain = dict(ops)
    plain["W3"] = plain["b3"] = None
    assert_bitwise(dual_gated, _run(plain, chunk_g=1), what="dual output with vs without the gate")


@requires_sm90
@pytest.mark.parametrize("N3", [128, 320, 64], ids=["exact_tile", "partial_tile", "narrow"])
@matrix_exempt(
    "sweeps the output gate's WIDTH, which is not a matrix axis: the axis is whether the gate "
    "exists, and its width is a shape. N3=192 against a 128-wide work tile is the partial-last-tile "
    "case, which is where the store's predicate is the only thing stopping an overrun"
)
def test_the_output_gate_matches_its_reference_including_a_partial_last_tile(N3):
    """The gate's own result is correct, and its partial last tile does not overrun.

    N3=320 against a 256-wide work tile pads the appended region to 512, so the second gate tile is
    a quarter real and three quarters padding. ``N3 = D`` (128) is the workflow's own value. The padding rows are zero-filled weights, so their accumulator is zero and the
    sigmoid of the bias would be a perfectly plausible value to store -- which is why the store is
    predicated to the true extent rather than left to produce harmless-looking numbers.
    """
    ops = _build(N=128, K=_K, bias="both", N3=N3)
    _, gate3 = _run(ops, tile_N=256, chunk_g=1)
    assert gate3.shape == (_M, N3)
    _assert_close_gate3(gate3, ops)


# ------------------------------------------------------------------ the declared refusals


@requires_sm90
@LN_DUAL_GATED.parametrize_unsupported(
    "ab_dtype",
    "chunk_g",
    "bias",
    "gate3",
    "fusion_variant",
    "arg_fault",
    "tile_M",
    "pingpong",
    "x_gate",
)
def test_unsupported_combos_raise(
    ab_dtype,
    chunk_g,
    bias,
    gate3,
    fusion_variant,
    arg_fault,
    tile_M,
    pingpong,
    x_gate,
    expected_error,
    expected_match,
):
    """Every declared region is refused by the kernel's OWN front door, with a message naming the fix.

    ``pytest.raises`` rather than ``xfail``: an xfail'd correctness assertion cannot tell "refused"
    from "answered wrongly", so a silently-wrong kernel would satisfy it and the suite would stay
    green.
    """
    ops = _apply_arg_fault(
        _build(dtype=ab_dtype, bias=bias, N3=_K if gate3 else 0, x_gate=x_gate), arg_fault
    )
    with front_door_raises(expected_error, expected_match):
        _run(
            ops,
            chunk_g=chunk_g,
            fusion_variant=fusion_variant,
            tile_M=tile_M,
            pingpong=pingpong,
        )


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "ab_dtype",
    "ln_bias",
    "bias",
    "mask",
    "fusion_variant",
    drop={"ab_dtype": (torch.float32,), "bias": ("gate_only",)},
    because=(
        "the dropped values are exactly the declared unsupported regions, which "
        "test_unsupported_combos_raise sweeps instead"
    ),
)
def test_the_two_activation_gate_reads_its_own_input_and_the_value_keeps_the_layernorm(
    ab_dtype, ln_bias, bias, mask, fusion_variant
):
    """The two-A path matches a reference where ONLY the value arm is normalized.

    **The load-bearing part is that ``x_gate`` is drawn independently of ``x``.** Every other
    correctness cell in this module feeds one activation to both projections, so a kernel that
    routed the wrong tensor into the gate would still agree with the reference. Here it cannot: the
    gate's input shares nothing with the value's, so feeding it ``LN(x)`` -- the single most
    plausible mis-wiring, since that is what every other variant does -- moves the output by
    O(1), not by an ulp.

    **Both fusions are swept, and each is held to its OWN bound.** They reach the same value by
    different arithmetic -- one folds the gain into the weight and repairs the accumulator, the
    other rewrites the staged activation and rounds it to 16 bits -- so the bounds are not
    interchangeable in either direction. On the physical fusion the mis-wiring this test is named
    for is a DIFFERENT mistake than on the fold: there, the gate's own staged tile would have to be
    normalized in place beside the value's, and nothing about its shape or dtype would object.

    Crossing ``ln_bias`` matters on both, for different reasons: on the fold it becomes the separate
    epilogue operand ``d = b_ln @ Wp``, on the physical fusion the normalize's add -- and in both
    cases its ABSENCE is a compile key rather than a zero operand.
    """
    ops = _build(dtype=ab_dtype, ln_bias=ln_bias, bias=bias, mask=mask, x_gate=True)
    got = _run(ops, fusion_variant=fusion_variant)
    _assert_close(got, ops, what="x_gate dual", fusion_variant=fusion_variant)


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "transpose_out",
    "tile_M",
    "fusion_variant",
    drop={"tile_M": (320,)},
    because="320 is the declared unsupported tile_M, which has no correct output to compare against",
)
def test_the_two_activation_path_tiles_and_transposes_like_the_one_activation_one(
    transpose_out, tile_M, fusion_variant
):
    """Every CTA tile M and both store orientations, on both two-A fusions.

    Not covered by the one-activation sweep: this kernel has its OWN epilogue -- it threads two
    accumulators as locals rather than going through the shared subtile loop -- so the store
    orientation and the epilogue subtile count are re-derived here and can be wrong here alone.
    Both fusions share that epilogue through the two-A mixin, so a break in it would show on
    either; what they do NOT share is the mainloop, and ``tile_M`` sizes the statistics scratch
    both mainloops publish through.
    """
    ops = _build(dtype=torch.bfloat16, x_gate=True)
    got = _run(ops, fusion_variant=fusion_variant, tile_M=tile_M, transpose_out=transpose_out)
    _assert_close(got, ops, what="x_gate dual", fusion_variant=fusion_variant)


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "M",
    "N",
    "K",
    "fusion_variant",
    drop={"M": (262144, 1048576, 4194304), "N": (64,), "K": ()},
    because=(
        "the three largest M are the workflow perf ladder and cost minutes apiece at this cell "
        "count; N=64 is dropped because it is below the 128 the pool's smallest workflow width "
        "uses and adds no path this axis does not already reach"
    ),
)
def test_the_two_activation_path_accepts_off_grid_and_odd_extents(M, N, K, fusion_variant):
    """Off-grid tokens, an odd feature width and a K no k-tile divides, on both two-A fusions.

    ``N=137`` is the one that earns its place: on this path ``tile_N`` tiles the OUTPUT width
    directly rather than the 2N pre-activation, so the tile arithmetic is not the dual's and an
    odd N reaches a different partial-tile predicate. ``K=136`` reaches the padded contraction,
    where the TMA zero-fills the last k-tile -- and the row statistics must still divide by the TRUE
    K, which is the one place a padded contraction can silently change a LayerNorm.

    **The padded contraction is where the two fusions differ most**, which is why both are swept
    here rather than only the fold: the physical fusion reads the padded tail through the STAGED
    ``(K,)`` gain, whose ``[K_real, K_gemm)`` slots are zeroed so the zero-filled activation column
    maps to exactly zero, while the fold never touches a gain at all. A padded K is the one shape
    that can tell them apart.

    The tile is chosen WITHOUT requiring that it divide N, which is the whole point: this path adds
    no divisibility constraint, and a picker that searched for a dividing tile would quietly stop
    testing the partial last tile at exactly the extents that have one.
    """
    tile_N = min(128, (N + 15) // 16 * 16)
    ops = _build(M=M, N=N, K=K, dtype=torch.bfloat16, x_gate=True)
    got = _run(ops, fusion_variant=fusion_variant, tile_N=tile_N)
    _assert_close(got, ops, what="x_gate dual", fusion_variant=fusion_variant)


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "N",
    "fusion_variant",
    only={"N": (320,)},
    because=(
        "the `narrow_dividing_tile` facet is the whole subject: this asserts the statistics "
        "reduction is not racing the stage refill, and the race is only reachable where the "
        "largest dividing tile is narrower than 128. 320 is the pool's only such value"
    ),
)
def test_the_two_activation_stats_survive_a_narrow_dividing_tile(N, fusion_variant):
    """The same operands must give the same answer twice, at the tile that recycles stages fastest.

    **This is a regression test for a silent wrong answer, so it asserts STABILITY first.** The
    kernel reads each staged A tile twice -- once by the WGMMA, once by the row-statistics reduction
    -- and only the WGMMA's read is drained by ``wait_group(0)``. Without the closing fence in
    `accumulate_row_stats` the stage is handed back to the TMA producer with the reduction's ``LDS``
    still in flight, and it returns the NEXT k-tile's activations. Max observed error 0.16-0.30, far
    above rounding, but the answer is plausible and self-consistent, which is why nothing caught it.

    Three things have to be true at once or this test measures nothing, and each was a blind spot:

    * ``tile_N`` must be the DIVIDING tile (64 here), not the ``min(128, ...)`` the neighbouring
      off-grid test uses -- at 128 this path is clean, so the pre-existing coverage sat exactly
      beside the defect. A narrower tile means more N-tiles per work tile, so the store warp spends
      longer in the epilogue and re-enters the mainloop last.
    * ``M`` must be large enough to give the scheduler many work tiles; the defect is invisible at
      the pool's small M.
    * It must run TWICE and compare bitwise. A single run against the reference passes: the
      corrupted statistics are wrong by less than a loose tolerance on most rows.

    N, not K, selects the regime -- measured 17/20 and 20/20 corrupt at N=320 with K=320 and 384,
    against 0/20 at N=384 with the same two K.

    **Both fusions are swept, and the physical one is the more interesting of the two.** Its stats
    pass reads the staged tile with NO WGMMA alongside, so there is not even a ``wait_group`` in
    front of the release -- structurally MORE exposed than the fold's, which is why it is asserted
    here rather than assumed clean. It measures 0/40 unfenced at this shape (the fold measures
    40/40), so it ships without the fold's fence; this test is what would notice if that stopped
    being true.
    """
    ops = _build(M=32768, N=N, K=N, dtype=torch.bfloat16, x_gate=True)
    tile_N = max(v for v in range(32, 129, 32) if N % v == 0)
    assert tile_N < 128, f"N={N} must have a dividing tile narrower than 128; got {tile_N}"
    # A race is a RATE, so one pair is a coin flip, not a test. Unfixed, the FOLD cell corrupts ~72%
    # of pairs (29/40 measured, 95% CI [56, 85]), so a single comparison would go green on more than
    # one run in four and read as a fixed kernel. Five pairs put a miss under 1 in 500. Kept small
    # because each pair is two launches on an already-compiled kernel -- the whole test is ~7 s.
    first = _run(ops, fusion_variant=fusion_variant, tile_N=tile_N)
    for attempt in range(5):
        other = _run(ops, fusion_variant=fusion_variant, tile_N=tile_N)
        n_bad = int((first != other).sum())
        assert torch.equal(first, other), (
            f"the two-A {fusion_variant} path is not deterministic at N={N}, tile_N={tile_N}: on "
            f"repeat {attempt + 1} of 5, {n_bad} of {first.numel()} elements differ from the first "
            f"run on identical operands, worst "
            f"{float((first.float() - other.float()).abs().max()):.4f}. That is the statistics "
            f"reduction racing the stage refill, not rounding."
        )
    _assert_close(first, ops, what="x_gate dual", fusion_variant=fusion_variant)


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "tile_M",
    "fusion_variant",
    drop={"tile_M": (320,)},
    because="320 is the declared unsupported tile_M, which has no correct output to compare against",
)
def test_the_two_activation_path_accepts_an_mn_major_value_activation(tile_M, fusion_variant):
    """An MN-major value ``x`` beside a K-major ``x_gate`` -- the majors the TriMul back half feeds.

    **This is the shape the workflow actually runs**, and it was uncovered until this test: every
    other cell in this file builds a contiguous, K-major activation, so the whole differing-majors
    branch -- the gate's own MMA atom, its own staged ``sA2`` layout, and its own TMA descriptor --
    was reached by nothing.

    **On the physical fusion it also selects a DIFFERENT MAINLOOP.** An MN-major staged tile is
    strided along K, and that fusion reads it twice, so it moves the tile into the WGMMA's operand
    registers and does both the reduction and the normalize there. ``tile_M`` is the axis that
    matters for it: the register layout's row mapping is closed-form for a 128-row CTA tile, so 128
    takes the register mainloop and 64 and 256 fall back to the shared-memory one. Both must be
    correct, and only a sweep over ``tile_M`` at an MN-major ``x`` says so -- a single tile would
    test one of the two and look like it tested the variant.

    The algebraic fold reads the tile once, so it stays on the shared-memory mainloop at every
    tile_M and this is a pure differing-majors test for it.
    """
    ops = _build(dtype=torch.bfloat16, x_gate=True, bias="both")
    # An MN-major VIEW of the same values: a (K, M) buffer transposed, so the K stride is M. The
    # values are unchanged, which is what lets the shared reference and bound apply unaltered.
    ops["x"] = ops["x"].T.contiguous().T
    assert ops["x"].stride(1) == ops["x"].shape[0], (
        "x must be MN-major for this test to mean anything"
    )
    got = _run(ops, fusion_variant=fusion_variant, tile_M=tile_M)
    _assert_close(got, ops, what="x_gate dual, MN-major x", fusion_variant=fusion_variant)


@requires_sm90
@matrix_exempt("checks that two ARGUMENTS are distinct, which is not a matrix axis")
def test_the_two_activations_must_be_different_tensors():
    """Passing one tensor as both is refused rather than computed.

    It IS expressible -- a gate over the raw activation while the value normalizes -- but no caller
    means it, and its output is indistinguishable from a wiring mistake. Refusing at the door is
    what makes the mistake findable.
    """
    ops = _build(x_gate=True, bias="both")
    ops["x_gate"] = ops["x"]
    with pytest.raises(ValueError, match=r"DIFFERENT tensor"):
        _run(ops, fusion_variant="alg_fold")


@requires_sm90
@pytest.mark.parametrize("fusion_variant", FUSION_VARIANTS)
@pytest.mark.parametrize("tile_N", [16, 48, 128, 2 * _N], ids=["narrow", "off16", "exact", "over"])
@matrix_exempt("sweeps tile_N against a FIXED N, which is a relation between two axes and not one")
def test_the_two_activation_path_adds_no_divisibility_constraint_on_the_output_width(
    tile_N, fusion_variant
):
    """Any ``tile_N`` is correct, including ones that do not divide N and one LARGER than it.

    **This test exists because the front door briefly refused these, and the refusal was wrong.**
    ``tile_N`` does mean something different here -- it tiles the output width directly, not the 2N
    pre-activation -- and the upstream's auto-pick searches for a tile that divides N, which reads
    like a requirement. It is not one: the partial last tile is predicated exactly as on the dual
    path, which has always accepted a tile that does not divide its 2N. Adding the constraint would
    have been a regression against this package's own base kernel, not a feature of the variant.

    ``over`` (``tile_N = 2N``) is the case a caller reaches by moving a working dual call across
    without halving the tile. It is wasteful -- half of every tile is padding -- and correct, which
    is the right pair: a performance mistake should not present as a shape error.
    """
    ops = _build(x_gate=True, bias="both")
    got = _run(ops, fusion_variant=fusion_variant, tile_N=tile_N)
    _assert_close(got, ops, what="x_gate dual", fusion_variant=fusion_variant)


@requires_sm90
@matrix_exempt("checks a SHAPE precondition of the output gate, not a matrix axis")
def test_an_output_gate_whose_region_boundary_is_off_tile_is_refused():
    """``2N`` must fall on a work-tile edge, or the two regions would share a tile.

    A shared tile is the one case the runtime region test cannot express: the branch is per work
    tile, so a tile straddling the boundary would take one arm for columns belonging to the other.
    """
    ops = _build(N=96, K=_K, bias="both", N3=_K)
    with pytest.raises(ValueError, match=r"must be a multiple of tile_N"):
        _run(ops, tile_N=_TILE_N, chunk_g=1)


@requires_sm90
@matrix_exempt("checks the pairing of two arguments, which is not a matrix axis")
def test_the_output_gates_weight_and_destination_must_be_given_together():
    """One without the other is a caller error with no sensible interpretation."""
    ops = _build(N3=_K, bias="both")
    dual = _pitch_pad(_M, _N, torch.bfloat16)
    with pytest.raises(ValueError, match=r"must be given together"):
        layernorm_dual_gated_gemm(
            ops["x"],
            ops["norm_weight"],
            ops["Wg"],
            ops["Wp"],
            dual,
            _TILE_M,
            _TILE_N,
            W3=ops["W3"],
            b3=ops["b3"],
            PostAct3=None,
        )


@requires_sm90
@matrix_exempt("checks a dtype precondition of one operand, which is not a matrix axis")
def test_a_sixteen_bit_layernorm_gain_is_refused_rather_than_silently_converted():
    """The gain is applied in fp32; a 16-bit one would lose more precision than the normalize itself."""
    ops = _build()
    ops["norm_weight"] = ops["norm_weight"].to(torch.bfloat16)
    with pytest.raises(ValueError, match=r"norm_weight must be torch.float32"):
        _run(ops, chunk_g=1)


@requires_sm90
@matrix_exempt("checks the package's single shape constraint, which is not a matrix axis")
def test_a_contraction_extent_below_the_alignment_floor_is_named_at_the_door():
    """The 16-byte floor is the ONLY shape constraint, and violating it is reported as such."""
    ops = _build(K=132)
    with pytest.raises(ValueError, match=r"16-byte alignment floor"):
        _run(ops, chunk_g=1)


# ------------------------------------------------------------------ structural gates


@matrix_exempt("audits the parameter pack, with no kernel launch")
def test_the_pack_extends_the_unfused_one_rather_than_replacing_it():
    """The fused pack must CONTAIN the dual-gated pack's fields, or the base is left half-configured.

    ``_bind_params`` binds exactly one pack. A subclass pack that dropped a base field would leave
    it unbound and the functor would be configured for a kernel it is not building.
    """
    base = set(DualGatedGemmParams.__annotations__) | {
        f for c in DualGatedGemmParams.__mro__ for f in getattr(c, "__annotations__", {})
    }
    fused = {
        f for c in LayerNormDualGatedGemmParams.__mro__ for f in getattr(c, "__annotations__", {})
    }
    assert base <= fused, f"the fused pack dropped {sorted(base - fused)}"
    for added in ("fusion_variant", "blk_k", "gemm_k", "gemm_k_real"):
        assert added in LayerNormDualGatedGemmParams.__annotations__, (
            f"{added} must be DECLARED on the fused pack, not stashed on the instance -- a value "
            f"folded into a compiled kernel but reassignable afterwards is exactly what the pack "
            f"mechanism exists to prevent"
        )
    # The output gate's fields belong to the PARENT pack and must not be re-declared here: the gate
    # is a property of the dual-gated GEMM, not of this fusion, and a shadowing re-declaration would
    # give the subclass a second field of the same name whose value the parent never sees.
    for owned_above in ("chunk_g", "n_dual_tiles", "gate3_n3"):
        assert owned_above not in LayerNormDualGatedGemmParams.__annotations__, (
            f"{owned_above} is DualGatedGemmParams' field; re-declaring it here shadows the "
            f"parent's and splits one parameter into two."
        )
        assert owned_above in DualGatedGemmParams.__annotations__


@matrix_exempt("audits the class's declared seams, with no kernel launch")
def test_the_fusion_overrides_only_the_two_declared_mainloop_seams():
    """The fusion must not re-implement the warpgroup role, the kernel entry or the epilogue loop.

    The whole argument for this design is that a fused mainloop is two hook overrides plus free
    functions. If ``kernel``, ``mma_warpgroup_role`` or ``producer_warpgroup_role`` ever appear
    here, that argument has quietly stopped being true -- and a re-implemented role drifts from the
    base it copied, which is the failure this file exists to prevent.
    """
    forbidden = {"kernel", "mma_warpgroup_role", "producer_warpgroup_role", "epilogue", "__call__"}
    overridden = forbidden & set(LayerNormDualGatedGemmSm90.__dict__)
    assert not overridden, (
        f"the fused class re-implements {sorted(overridden)}. Factor an extract-method hook into "
        f"the base instead, so the local default stays byte-identical."
    )


@matrix_exempt("audits the module's source, with no kernel launch")
def test_the_mainloop_seams_are_plain_defs_because_the_dsl_cannot_flatten_an_object():
    """``mma_setup_fragments`` and ``mma_consume_work_tile`` must NOT carry ``@cute.jit``.

    Both take or return an ``MmaFragments``, and ``@cute.jit`` flattens every argument into MLIR
    values -- a decorated override fails with ``DSLTreeFlattenError`` at trace time, which is a
    confusing place to learn a signature rule. Asserted here so the rule is checked where it is
    written down.
    """
    src = inspect.getsource(LayerNormDualGatedGemmSm90)
    tree = ast.parse("class C:\n" + "\n".join("    " + line for line in src.splitlines()[1:]))
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in (
            "mma_setup_fragments",
            "mma_consume_work_tile",
        ):
            names = [
                d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", "")
                for d in fn.decorator_list
            ]
            assert "jit" not in names, f"{fn.name} must be a plain def; it carries {names}"


@matrix_exempt("audits the matrix itself")
def test_the_matrix_is_diverse():
    """Every facet of every axis is straddled by its pool, above the entropy floor."""
    from fold_cp_ops.testing.kernel_matrix import MIN_FACET_ENTROPY

    weak = [
        s
        for axis in LN_DUAL_GATED.axes
        for s in axis.diversity()
        if s.waived is None and s.entropy < MIN_FACET_ENTROPY
    ]
    assert not weak, "under-covered facets:\n  " + "\n  ".join(str(s) for s in weak)


#: The value sets the correctness tests actually run, ONE per kernel shape rather than one overall.
#: An axis is pinned wherever a value is supported only in combination -- ``pingpong=True`` needs
#: ``alg_fold``, ``x_gate=True`` needs ``chunk_g=1`` AND no output gate -- so a single cross product
#: would have to drop those values entirely to stay region-free, which is how a whole kernel stops
#: being checked against the regions at all.
#:
#: ``fusion_variant`` is NOT pinned in the two-activation set, and that is the shape of the change
#: that landed the physical two-A kernel: both fusions have a two-A form, so both are run.
_SUPPORTED_SETS = {
    "one_activation": {
        "ab_dtype": (torch.bfloat16, torch.float16),
        "chunk_g": (1, 16),
        "bias": ("none", "both"),
        "gate3": (False,),
        "arg_fault": ("none",),
        "fusion_variant": ("prolog_ln", "alg_fold"),
        "tile_M": (64, 128, 256),
        "pingpong": (False,),
        "x_gate": (False,),
    },
    "two_activation": {
        "ab_dtype": (torch.bfloat16, torch.float16),
        "chunk_g": (1,),
        "bias": ("none", "both"),
        "gate3": (False,),
        "arg_fault": ("none",),
        "fusion_variant": ("prolog_ln", "alg_fold"),
        "tile_M": (64, 128, 256),
        "pingpong": (False,),
        "x_gate": (True,),
    },
}


@matrix_exempt("audits the matrix itself")
@pytest.mark.parametrize("shape", sorted(_SUPPORTED_SETS))
def test_every_supported_cell_is_outside_the_unsupported_regions(shape):
    """The values the correctness tests run must not land in a region declared to raise."""
    supported = _SUPPORTED_SETS[shape]
    for combo in itertools.product(*supported.values()):
        bound = dict(zip(supported, combo))
        for region in LN_DUAL_GATED.regions():
            assert not region.where(**{k: bound[k] for k in region.axis_names()}), (
                f"{bound} is run as a supported {shape} cell but lands in the region "
                f"{region.reason!r}"
            )


@matrix_exempt("audits the matrix itself")
def test_the_k_tile_pool_is_exactly_what_the_layout_module_declares():
    """The axis must not drift from the pool the front door and the functor validate against.

    A tile in the test pool but not in the module's would be refused at construction and read as a
    kernel bug; one in the module's but not the test's would ship unexercised.
    """
    assert tuple(LN_DUAL_GATED.axis("blk_k").values) == tuple(ALLOWED_BLK_K)
    assert auto_blk_k(_K) in ALLOWED_BLK_K


@requires_sm90
@pytest.mark.parametrize(
    "tile_M,tile_N,K",
    [(64, 32, 16), (64, 64, 16), (128, 128, 128)],
    ids=["deepest", "deep", "normal"],
)
@matrix_exempt(
    "sweeps TILE GEOMETRY against the shared-memory budget, which is not a matrix axis: the tiles "
    "here are chosen to drive the mainloop pipeline DEPTH, and the deepest is a geometry no "
    "workflow uses but the front door accepts"
)
def test_a_tile_small_enough_to_ask_for_a_very_deep_pipeline_still_launches(tile_M, tile_N, K):
    """A tiny tile asks for a huge pipeline; the shared-storage struct must still fit.

    ``GemmSm90._compute_stages`` budgets a FLAT 1024 bytes for every mbarrier array and alignment
    pad, and the mainloop pipeline's barriers cost 16 bytes a stage -- so past 64 stages the struct
    is bigger than what was budgeted. The unfused kernel lands just under the cap and survives;
    this one adds three shared buffers and tips over. Measured before the fix: a 233472-byte struct
    against a 232448-byte cap, and the symptom was ``cudaErrorInvalidValue`` at LAUNCH -- no compile
    error, no mention of shared memory, and nothing wrong with the arithmetic.

    Gated at all three depths rather than only the failing one, because the fix is a ceiling and a
    ceiling set too low would silently shorten the pipeline at shapes that need it.
    """
    ops = _build(M=128, N=32, K=K, bias="both")
    _assert_close(_run(ops, tile_M=tile_M, tile_N=tile_N, chunk_g=1), ops)


#: Cold-compile seconds measured in a FRESH process against ``main``'s ``staged`` kernel -- the one
#: this reproduces -- at the same shape, tile, layout and terms. Both columns run with the disk
#: cache off; a warm cache would only HIDE the number.
#:
#:   shape / tile_N / chunk_g / gate3        main    here
#:   4096x256x128  256  cg16  --             3.74    3.74
#:   4096x128x128  128  cg16  --             3.09    2.77
#:   4096x256x256  256  cg1   gate3          5.97    5.29
#:   4096x256x384  256  cg1   --             6.11    5.70
#:   4096x512x512  256  cg16  --             8.19    7.87
#:
#: Three of the five are OVER the package's 5 s bar, and they are over on both sides. The cause is
#: the one CLAUDE.md names: the consumer's two k-loops are ``range_constexpr``, so the normalize and
#: the WGMMA chain are emitted once per k-tile per pass, and the count grows with K. That is a
#: deliberate trade recorded in ``mma_prolog_ln`` -- a runtime loop compiles fast and runs 7-20%
#: slower -- and it is the same trade the kernel being reproduced makes. What this test holds is the
#: obligation that comes with the package's known-inherited-defect note: **do not make it worse.**
_COLD_COMPILE_CEILING_S = {(128, 16, False): 5.0, (384, 1, False): 6.11, (512, 16, False): 8.19}


@requires_sm90
@pytest.mark.parametrize(
    "K,chunk_g,gate3", list(_COLD_COMPILE_CEILING_S), ids=["k128_at_the_bar", "k384", "k512"]
)
@matrix_exempt(
    "times a COMPILE rather than sweeping configurations: the axes here are chosen for their effect "
    "on the emitted instruction count, and each cell carries its own measured ceiling"
)
def test_the_cold_compile_does_not_regress_past_what_it_reproduces(K, chunk_g, gate3, monkeypatch):
    """Cold compile, in a FRESH interpreter, against the ceiling measured on the kernel reproduced.

    At K=128 -- the TriMul feature width the workflow actually runs -- the ceiling IS the package's
    5 s bar and the cell is compliant. At K=384 and K=512 the ceiling is ``main``'s own measured
    number, because both sides breach the bar for the same reason and closing it is separate,
    scoped work; see the table above.

    Run in a subprocess because "cold" means cold: an in-session measurement is polluted by every
    MLIR context and cutlass import the suite has already paid for, which is worth several seconds
    and would let a real regression hide underneath it.
    """
    import subprocess
    import sys
    import textwrap

    src = textwrap.dedent(f"""
        import os, time, torch
        os.environ["CPO_CACHE_ENABLED"] = "0"
        import fold_cp_ops._internal.cache_utils as cu
        cu.COMPILE_ONLY = True
        from fold_cp_ops.kernels.layernorm_dual_gated_gemm import layernorm_dual_gated_gemm
        K, cg, g3 = {K}, {chunk_g}, {gate3}
        M, N, tn = 4096, 256, 256
        dt = torch.bfloat16
        z = lambda *s, d=dt: torch.zeros(*s, device="cuda", dtype=d)
        kw = {{}}
        if g3:
            kw = dict(W3=z(K, K), b3=z(K, d=torch.float32), PostAct3=z(M, K))
        t0 = time.perf_counter()
        layernorm_dual_gated_gemm(
            z(M, K), z(K, d=torch.float32), z(N, K), z(N, K), z(M, N), 128, tn,
            norm_bias=z(K, d=torch.float32), bg=z(N), bp=z(N), mask=z(M), chunk_g=cg, **kw)
        print(time.perf_counter() - t0)
    """)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    elapsed = float(r.stdout.strip().splitlines()[-1])
    ceiling = _COLD_COMPILE_CEILING_S[(K, chunk_g, gate3)]
    assert elapsed <= ceiling * 1.15, (
        f"cold compile took {elapsed:.2f}s against a {ceiling:.2f}s ceiling. The known root-cause "
        f"class is a nested `if` inside a `cutlass.range_constexpr` loop -- and this kernel HAS "
        f"such loops by design (mma_prolog_ln), so check whether their bodies grew rather than "
        f"whether they exist."
    )


# ------------------------------------------------------- the declared shape axes, exercised
#
# Three tests rather than one product: the audit requires every FACET of every axis to be reached by
# the module, not every combination. K is swept in its own test because K is the extent this kernel
# BAKES -- the staged gain is a shared-memory allocation -- so each value costs a cold compile.


@requires_sm90
@LN_DUAL_GATED.parametrize("M")
def test_every_token_extent_is_accepted(M):
    """M is free: partial-tile, off-grid and multi-wave token counts all produce correct output.

    The multi-wave end is the one that matters here: a persistent CTA reuses ONE statistics scratch
    across tiles, so tile n+1's finalize can overwrite mu/rstd while tile n's normalize still reads
    them. Below one tile per CTA that hazard is never exercised.
    """
    ops = _build(M=M, N=_N, K=_K, bias="both", mask=True)
    _assert_close(_run(ops, chunk_g=1), ops)


@requires_sm90
@LN_DUAL_GATED.parametrize("N")
def test_every_feature_extent_is_accepted(N):
    """N is free, including the off-grid width and every TriMul feature width."""
    ops = _build(M=1000, N=N, K=_K, bias="both", mask=True)
    _assert_close(_run(ops, tile_N=min(2 * N, 256), chunk_g=1), ops)


@requires_sm90
@LN_DUAL_GATED.parametrize("K")
def test_every_contraction_extent_is_accepted(K):
    """K is the LayerNorm axis AND the one extent carrying the 16-byte floor; every pool value works."""
    ops = _build(M=1000, N=_N, K=K, bias="both", mask=True)
    _assert_close(_run(ops, chunk_g=1), ops)


@requires_sm90
@LN_DUAL_GATED.parametrize("transpose_out")
def test_every_store_majorness_is_accepted(transpose_out):
    """The transposed store changes the descriptor and the atom, never the arithmetic."""
    ops = _build(bias="both", mask=True)
    _assert_close(_run(ops, chunk_g=1, transpose_out=transpose_out), ops)


# ── the algebraic fold's precompute ───────────────────────────────────────────────────────────
# `build_folded_dual_operands` is the B-side half of `alg_fold`: it interleaves, folds and reduces
# in ONE launch. What must hold is (a) the elementwise outputs are BITWISE what a separate-step
# build produces, (b) the two reductions land within fp32 summation-order slack of it, and (c) it
# really is one launch -- which is the entire reason the fusion is worth having.


def _fold_case(N, K, dtype=torch.bfloat16, ln_bias=True, bias="both", chunk_g=1, seed=0):
    """Build one keyword set for `build_folded_dual_operands` and its reference.

    Args:
        N: Projection width per half. With ``chunk_g > 1`` it must be a multiple of `chunk_g`, which
            the caller arranges; this helper does not adjust it.
        K: Contraction extent, and the LayerNorm width. Must be a multiple of 8 (the 16-byte floor
            at 16 bits) or the front door refuses it.
        dtype: Weight dtype. fp32 is not a legal weight dtype for this kernel and is not offered.
        ln_bias: Whether to build the ``(K,)`` LayerNorm bias, and hence the `dbias` output.
        bias: ``"none"``, ``"both"``, or ``"gate_only"`` -- the projection biases. ``"gate_only"``
            is legal here even though the dual GEMM's front door refuses it, because this kernel
            zero-fills the missing side rather than pairing them.
        seed: RNG seed, so a failure is reproducible.

    Returns:
        A kwargs dict accepted verbatim by both `build_folded_dual_operands` and its reference.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda *s, dt=dtype: torch.randn(*s, device="cuda", dtype=dt, generator=g)
    return dict(
        Wg=mk(N, K),
        Wp=mk(N, K),
        norm_weight=mk(K, dt=torch.float32),
        norm_bias=mk(K, dt=torch.float32) if ln_bias else None,
        bg=mk(N) if bias in ("both", "gate_only") else None,
        bp=mk(N) if bias == "both" else None,
        chunk_g=chunk_g,
    )


def _assert_fold_matches(kw, colsum_rounded=False):
    """Compare `build_folded_dual_operands` against its reference, each field by its own rule.

    Purpose
        The four outputs do not all admit the same comparison, and using one rule for all of them
        either fails spuriously or passes vacuously.

    Semantics
        `B` and `gated_bias` are pure elementwise work, so they must be BITWISE equal -- a fusion has
        no freedom there and a difference is a defect. `colsum` and `dbias` are reductions whose
        summation order the kernel fixes and torch does not, so they are compared against the fp32
        summation bound ``gamma_K * sum|terms|``, computed from the same operands. That bound is
        derived, not tuned: it is the standard ``K*u/(1-K*u)`` for a K-term fp32 sum, so a
        disagreement larger than it is a real difference and not an ordering artifact.

    Args:
        kw: The kwargs from :func:`_fold_case`, passed to both implementations unchanged.

    Returns:
        None.

    Raises:
        AssertionError: Naming the field that disagreed and, for the reductions, by how much
            relative to the allowance.
    """
    got = build_folded_dual_operands(**kw, colsum_rounded=colsum_rounded)
    want = build_folded_dual_operands_ref(**kw, colsum_rounded=colsum_rounded)
    assert_bitwise(got.B, want.B, what="folded weight B")
    if want.gated_bias is None:
        assert got.gated_bias is None
    else:
        assert_bitwise(got.gated_bias, want.gated_bias, what="interleaved projection bias")

    K = kw["Wg"].shape[1]
    u = 2.0**-24
    gamma = (K * u) / (1.0 - K * u)
    # The magnitudes each reduction runs over, in the interleaved column order the reference builds.
    ref_terms = (kw["norm_weight"].unsqueeze(0).float() * kw["Wg"].float()).abs().sum(1)
    ref_terms = torch.maximum(
        ref_terms, (kw["norm_weight"].unsqueeze(0).float() * kw["Wp"].float()).abs().sum(1)
    )
    allow = gamma * ref_terms.max() * 2.0  # x2: both implementations round, in different orders
    assert_elementwise(
        got.colsum, want.colsum, allow, what="colsum (bound is what fp32 summation ORDER allows)"
    )
    if want.dbias is None:
        assert got.dbias is None
    else:
        dterms = (kw["norm_bias"].unsqueeze(0).float() * kw["Wg"].float()).abs().sum(1)
        dterms = torch.maximum(
            dterms, (kw["norm_bias"].unsqueeze(0).float() * kw["Wp"].float()).abs().sum(1)
        )
        d_allow = gamma * dterms.max() * 2.0
        assert_elementwise(
            got.dbias,
            want.dbias,
            d_allow,
            what="dbias (bound is what fp32 summation ORDER allows)",
        )


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "ab_dtype",
    "ln_bias",
    "bias",
    drop={"ab_dtype": (torch.float32,)},
    because="the folded weight IS a GEMM operand, so fp32 has no kernel -- the front door raises",
)
def test_the_fold_precompute_matches_a_separate_step_build(ab_dtype, ln_bias, bias):
    """Every combination of the two optional bias terms reproduces the separate-step build."""
    _assert_fold_matches(_fold_case(128, 128, ab_dtype, ln_bias, bias))


@requires_sm90
@LN_DUAL_GATED.parametrize(
    "chunk_g",
    drop={"chunk_g": (8,)},
    because="8 is the declared-unsupported rung; the front door raises on it, swept below",
)
def test_the_fold_precompute_emits_both_interleave_orders(chunk_g):
    """Both interleave orders are reproduced, and reproduced consistently.

    An off-by-one in either mapping still produces a full, finite, plausibly-scaled tensor, so a
    tolerance-based check would not see it; the reference builds the column order from the rule
    independently, so a mapping error shows as a BITWISE difference in `B`.

    **Both orders are anchored outside this pair of functions, but only one of them by a test.**
    The element interleave matches `_interleave_dual_weights_kernel`, which writes ``p`` even as the
    gate, and that kernel is exercised end to end elsewhere in this module. The block interleave
    matches `GemmGatedMixin`: `maybe_override_epi_tile` forces ``epi_n = 2*chunk_g``, and
    `epi_gate_preact` then computes ``out[j] = act_fn(rD[j+H], rD[j])`` with ``H = chunk_g`` -- so
    registers ``[0, G)`` are up and ``[G, 2G)`` are gate, which is exactly the mapping the fold
    emits. That agreement is established by READING those two methods, not by running anything: no
    existing path materializes a ``(2N, K)`` operand for ``chunk_g > 1``, so there is nothing to
    compare the fold's block output against until `alg_fold`'s epilogue consumes it. A block order
    that disagreed with `epi_gate_preact` would still pass here.
    """
    _assert_fold_matches(_fold_case(128, 128, chunk_g=chunk_g))


@requires_sm90
@LN_DUAL_GATED.parametrize("N")
def test_the_fold_precompute_accepts_every_feature_extent(N):
    """N is free, including the off-grid 137 whose ``2N`` does not tile the 64-wide column tile.

    That case is the one the compile-time store guard exists for: with ``2N = 274`` the last column
    tile is partial, and an unguarded store would write 54 columns past the end of `B`. It is in the
    declared pool precisely so this cannot be quietly dropped.
    """
    _assert_fold_matches(_fold_case(N, 128, chunk_g=1))


@requires_sm90
@LN_DUAL_GATED.parametrize("K")
def test_the_fold_precompute_accepts_every_contraction_extent(K):
    """K is the reduction axis; every pool width folds and reduces correctly."""
    _assert_fold_matches(_fold_case(128, K, chunk_g=1))


@requires_sm90
@matrix_exempt("asserts a launch COUNT, which no shape or dtype axis varies")
def test_the_fold_precompute_costs_one_device_launch():
    """The whole B-side build is ONE kernel -- the claim the algebraic fusion rests on.

    Separate steps would be an interleave, a fold, two reductions and a bias interleave, plus a
    widening cast per 16-bit bias vector. The count is asserted rather than the time because it is
    exact: a regression here is someone reaching for a ``.float()`` or a ``.contiguous()``, which a
    timing test would absorb into noise.
    """
    kw = _fold_case(128, 128, bias="both")
    kernels = _device_kernels(lambda: build_folded_dual_operands(**kw))
    assert len(kernels) == 1, f"the fold precompute must be one fused launch; got {kernels}"
    assert "fold_precompute" in kernels[0], f"and it must be THE fold kernel; got {kernels}"


@requires_sm90
@matrix_exempt("sweeps malformed arguments, whose whole point is to fall outside the declared pool")
@pytest.mark.parametrize(
    "bad,match",
    [
        ("Wg_fp32", "16-bit"),
        ("Wp_shape", "Wp"),
        ("norm_weight_dtype", "norm_weight"),
        ("norm_weight_shape", "norm_weight"),
        ("norm_bias_dtype", "norm_bias"),
        ("bg_shape", "bg"),
        ("chunk_g_odd", "chunk_g"),
        ("chunk_g_indivisible", "chunk_g"),
        ("K_unaligned", "Wg"),
    ],
)
def test_the_fold_precompute_refuses_a_malformed_argument(bad, match):
    """Every argument is held to its contract at the front door, by name.

    The alternative is not a clean failure further in: a wrong-dtype gain would be read as fp32
    garbage, and an indivisible ``chunk_g`` maps two output columns onto one weight row and drops the
    other -- both produce a full, finite, wrong tensor.
    """
    kw = _fold_case(128, 128, bias="both")
    if bad == "Wg_fp32":
        kw["Wg"] = kw["Wg"].float()
    elif bad == "Wp_shape":
        kw["Wp"] = kw["Wp"][:64]
    elif bad == "norm_weight_dtype":
        kw["norm_weight"] = kw["norm_weight"].bfloat16()
    elif bad == "norm_weight_shape":
        kw["norm_weight"] = kw["norm_weight"][:64]
    elif bad == "norm_bias_dtype":
        kw["norm_bias"] = kw["norm_bias"].bfloat16()
    elif bad == "bg_shape":
        kw["bg"] = kw["bg"][:64]
    elif bad == "chunk_g_odd":
        kw["chunk_g"] = 3
    elif bad == "chunk_g_indivisible":
        kw["chunk_g"] = 48  # a multiple of 16 that does not divide N = 128
    elif bad == "K_unaligned":
        kw = _fold_case(128, 132, bias="both")  # 132 % 8 != 0 at 16 bits
    with pytest.raises(ValueError, match=match):
        build_folded_dual_operands(**kw)


@requires_sm90
@matrix_exempt("asserts run-to-run reproducibility of one cell, not coverage of a pool")
def test_the_fold_precompute_reduction_is_bit_reproducible():
    """Two runs of the same input give bit-identical reductions, at a shape with a partial K tile.

    The fixed ascending order is what makes `colsum` comparable bitwise against another machine or
    another grid shape -- the property the whole `alg_fold` byte-identity gate will rest on. A
    partial last K tile is used because that is where a reduction is most likely to acquire an
    order-dependent branch.
    """
    kw = _fold_case(137, 136, bias="both")
    a = build_folded_dual_operands(**kw)
    b = build_folded_dual_operands(**kw)
    assert_bitwise(a.colsum, b.colsum, what="colsum")
    assert_bitwise(a.dbias, b.dbias, what="dbias")
    assert_bitwise(a.B, b.B, what="B")


@requires_sm90
@matrix_exempt("asserts an absence, which is a property of the None handling and not of a shape")
def test_an_absent_term_is_absent_rather_than_zero():
    """No LayerNorm bias means no `dbias` tensor, and no projection bias means no `gated_bias`.

    Returning zeros instead would be a quiet correctness cost, not a convenience: the epilogue would
    then add a term it could have compiled out, and `colsum`'s reduction would be joined by one whose
    fp32 rounding the absent path never incurs.
    """
    out = build_folded_dual_operands(**_fold_case(128, 128, ln_bias=False, bias="none"))
    assert out.dbias is None and out.gated_bias is None
    one_side = build_folded_dual_operands(**_fold_case(128, 128, bias="gate_only"))
    assert one_side.gated_bias is not None
    assert torch.equal(one_side.gated_bias[1::2], torch.zeros_like(one_side.gated_bias[1::2])), (
        "the absent up-projection bias must contribute zeros, the identity for a bias"
    )


@requires_sm90
@matrix_exempt("sweeps a reduction convention, which is not a shape or dtype axis")
@pytest.mark.parametrize("colsum_rounded", [False, True])
def test_both_colsum_conventions_are_reproduced(colsum_rounded):
    """Each convention matches its own reference. Both are needed; neither is a superset.

    The upstream uses BOTH -- unrounded in its two CUDA precompute kernels, rounded in its gate3
    torch helper -- so byte-identity with it is unreachable from a single convention. `False` is the
    default because that is what the dual path must reproduce; the gate3 port will pass `True`.
    """
    _assert_fold_matches(_fold_case(128, 128, bias="both"), colsum_rounded=colsum_rounded)


@requires_sm90
@matrix_exempt("asserts the flag changes exactly one output, which no shape axis varies")
def test_the_colsum_convention_moves_colsum_and_nothing_else():
    """The flag is live, and its blast radius is one tensor.

    Both halves matter. If `colsum` did not differ the flag would be a no-op that silently failed to
    give gate3 the convention it needs; if anything ELSE differed, switching conventions would change
    the folded weight the MMA reads, and the two could not be compared at all.
    """
    kw = _fold_case(128, 128, bias="both")
    a = build_folded_dual_operands(**kw, colsum_rounded=False)
    b = build_folded_dual_operands(**kw, colsum_rounded=True)
    assert not torch.equal(a.colsum, b.colsum), "the flag did nothing"
    assert_bitwise(a.B, b.B, what="folded weight B")
    assert_bitwise(a.dbias, b.dbias, what="dbias")
    assert_bitwise(a.gated_bias, b.gated_bias, what="interleaved projection bias")


# ── alg_fold: the algebraic variant's STRUCTURE ───────────────────────────────────────────────
# These are host-side. The DSL bodies are traced only when the front door routes to this class,
# which is the next step; until then a mistake inside `mma_consume_work_tile` or
# `epi_combine_preact` is NOT caught here, and saying so is the point of this comment.


@matrix_exempt("asserts class structure, which no shape or dtype axis varies")
def test_alg_fold_overrides_three_seams_and_no_more():
    """The fusion is five methods, each for a reason no other method can serve.

    `prolog_ln` is held to the same rule by
    `test_the_fusion_overrides_only_the_two_declared_mainloop_seams`; this is the algebraic
    variant's copy of it, and it is the check that a second fusion did not quietly fork the
    epilogue, the scheduler loop or the store the way both upstream fusions did.

    Every name below cost a real defect to learn, so the message says why each is not optional --
    the next person to "simplify" one of them should have to disagree with a measurement.
    """
    # `inspect.isfunction`, not `callable`: the metaclass GENERATES an `EpilogueParams` class from
    # `_epi_ops`, so it is callable and present in every functor's dict without being an override.
    declared = {
        m for m, v in vars(LayerNormDualGatedGemmAlgFoldSm90).items() if inspect.isfunction(v)
    }
    assert declared == {
        "load_AB",
        "mma_consume_work_tile",
        "epi_combine_preact",
        "epi_repair_rank_one",
        "epi_visit_subtile",
        "epi_get_smem_tensors",
    }, (
        f"alg_fold declares {sorted(declared)}; the fusion is the single-pass PRODUCER, the "
        f"mainloop consumer, the pre-activation combine, the shared repair, the region dispatch, "
        f"and the shared-memory hand-off.\n"
        f"  load_AB: the parent stages every k-tile TWICE for its two-pass sweep, and staging "
        f"twice against a one-pass consumer's waits is a HANG that compiles cleanly.\n"
        f"  epi_visit_subtile: the base's OUTPUT-GATE arm adds b3 and applies the activation with "
        f"no rank-one repair -- correct for a kernel that normalized in the prologue, and a "
        f"finite, plausible, entirely wrong gate for this one."
    )
    for forbidden in ("kernel", "mma_warpgroup_role", "producer_warpgroup_role", "epilogue"):
        assert forbidden not in vars(LayerNormDualGatedGemmAlgFoldSm90), (
            f"alg_fold overrides {forbidden}, which would fork the shared kernel body"
        )


@matrix_exempt("asserts the op-tuple's shape, which no shape or dtype axis varies")
def test_alg_fold_appends_its_epilogue_ops_and_moves_none():
    """Four ops appended, every inherited op at its original index.

    The order defines both the generated params struct and the shared-memory map, so an op MOVED is
    an op handed another op's buffer -- a wrong-answer condition with no fault. Appending is the only
    safe edit, and this is what keeps it that way.
    """
    parent = [op.name for op in LayerNormDualGatedGemmSm90._epi_ops]
    child = [op.name for op in LayerNormDualGatedGemmAlgFoldSm90._epi_ops]
    assert child[: len(parent)] == parent, "an inherited op moved"
    assert child[len(parent) :] == [
        "sRstd",
        "sScale",
        "mColsum",
        "mDbias",
        "mColsum3",
        "mDbias3",
    ], (
        "the gate's c3/d3 are SEPARATE ops, not a wider mColsum: `run_epilogue` re-bases the gate "
        "arm's n-block, so one shared vector is read at the DUAL region's columns on every gate "
        "tile -- a finite, plausible, entirely wrong gate output"
    )


@matrix_exempt("asserts which class builds which variant, not a kernel shape")
def test_each_class_implements_exactly_its_own_variant():
    """The refusal is keyed on the class, so neither variant can be built by the wrong one.

    This is what lets `alg_fold` be a legal `fusion_variant` value while only one fusion existed --
    and what keeps the parent refusing it now that the other one does exist.
    """
    assert LayerNormDualGatedGemmSm90._IMPLEMENTED_VARIANTS == ("prolog_ln",)
    assert LayerNormDualGatedGemmAlgFoldSm90._IMPLEMENTED_VARIANTS == ("alg_fold",)
    assert set(FUSION_VARIANTS) == {
        *LayerNormDualGatedGemmSm90._IMPLEMENTED_VARIANTS,
        *LayerNormDualGatedGemmAlgFoldSm90._IMPLEMENTED_VARIANTS,
    }, "a declared variant is built by no class, or a class builds an undeclared one"


@matrix_exempt("asserts class structure, which no shape or dtype axis varies")
def test_both_two_activation_classes_take_their_variant_from_their_fusion_base():
    """Each two-A class builds exactly one variant, and inherits WHICH from its fusion base.

    The two-A classes declare no ``_IMPLEMENTED_VARIANTS`` of their own, deliberately: the variant a
    class builds is a property of its FUSION, and re-declaring it here would let the two-A and
    one-A halves of one fusion drift into claiming different variants. `_xgate_functor_for` routes
    on exactly this attribute, so a drift would silently hand the front door the wrong class -- and
    the two classes' epilogue op tuples differ, so the symptom is a missing or surplus
    ``EpilogueArguments`` field at construction rather than a wrong number.
    """
    for two_a, one_a in (
        (LayerNormDualGatedGemmXGatePrologLnSm90, LayerNormDualGatedGemmSm90),
        (LayerNormDualGatedGemmXGateAlgFoldSm90, LayerNormDualGatedGemmAlgFoldSm90),
    ):
        assert "_IMPLEMENTED_VARIANTS" not in vars(two_a), (
            f"{two_a.__name__} re-declares _IMPLEMENTED_VARIANTS; it must inherit its fusion's"
        )
        assert two_a._IMPLEMENTED_VARIANTS == one_a._IMPLEMENTED_VARIANTS
        assert _xgate_functor_for(one_a._IMPLEMENTED_VARIANTS[0]) is two_a


@matrix_exempt("asserts class structure, which no shape or dtype axis varies")
def test_the_two_activation_mixin_holds_everything_the_two_fusions_share():
    """The shared two-A machinery lives on the mixin; each concrete class declares only its fusion.

    The whole argument for one mixin over two forked classes is that four operands per slot, the
    gate's own MMA atom, the halved pipeline and the two-accumulator epilogue are the SAME on both
    fusions -- so if one of those names reappears on a concrete class, two copies exist and the one
    that is not being read will drift. Upstream forks these classes across two modules and they
    HAVE drifted, which is what this is guarding against.
    """
    shared = {
        "a2_smem_layout_staged",
        "num_tma_load_bytes",
        "epi_convert_postact",
        "__call__",
        "kernel_xgate",
        "_producer_xgate",
        "_load_AB_xgate",
        "_consumer_xgate",
        "_epilogue_xgate",
        "_compute_stages",
        "_stage_ln_affine",
    }
    assert shared <= set(vars(_XGateTwoASm90)), (
        f"the mixin is missing {sorted(shared - set(vars(_XGateTwoASm90)))}"
    )
    # `_load_AB_xgate` is the ONE shared name a fusion may legitimately re-declare: the pass count
    # is the fusion's, and it must match its consumer's or the two deadlock.
    for cls in (LayerNormDualGatedGemmXGateAlgFoldSm90, LayerNormDualGatedGemmXGatePrologLnSm90):
        forked = (shared - {"_load_AB_xgate"}) & set(vars(cls))
        assert not forked, (
            f"{cls.__name__} re-declares {sorted(forked)}, which the mixin already owns -- two "
            f"copies of the two-A machinery is exactly what this design replaced"
        )


@matrix_exempt("asserts the producer/consumer sweep counts agree, which no shape axis varies")
def test_each_two_activation_fusion_stages_a_matched_number_of_sweeps():
    """``_A_SWEEPS`` must equal how many single-pass loads the class's producer actually issues.

    **A mismatch is a DEADLOCK, not a wrong answer** -- the producer fills the ring and blocks
    forever on an acquire nothing will release, and it is invisible to every host-side test because
    the kernel compiles perfectly and then never returns. The two facts live in different methods
    (the declared count, and the loop that implements it), so this is what ties them together.

    Counted from the SOURCE rather than by running the kernel, because the failure mode is a hang:
    a runtime check would be the thing that hangs.
    """
    for cls, expected in (
        (LayerNormDualGatedGemmXGateAlgFoldSm90, 1),
        (LayerNormDualGatedGemmXGatePrologLnSm90, 2),
    ):
        assert cls._A_SWEEPS == expected
        if "_load_AB_xgate" not in vars(cls):
            sweeps = 1  # inherits the mixin's single-pass body
        else:
            # Parse the CLASS source and find the def, rather than `getsource` on the attribute:
            # `@cute.jit` wraps it, and what `getsource` follows through the wrapper is not
            # something this test should depend on.
            tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
            fn = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_load_AB_xgate"
            )
            sweeps = sum(
                1
                for node in ast.walk(fn)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_load_AB_xgate"
            )
        assert sweeps == cls._A_SWEEPS, (
            f"{cls.__name__} declares _A_SWEEPS={cls._A_SWEEPS} but its producer issues {sweeps} "
            f"pass(es). The consumer waits _A_SWEEPS * k_tile_cnt times; a mismatch hangs."
        )


# ──────────────────────── config selection: the heuristic, the pool, the freeze ────────────
# Unit tests for `fold_cp_ops.kernels.layernorm_dual_gated_gemm_autotune`.
#
# **The arch-independent half is tested exhaustively and the arch-dependent half is not tested at
# all**, on purpose. The heuristic is one arch-keyed number (``PINGPONG_MAX_K``, bisected upstream on
# H200) wrapped in functional gates and a tileability check, and only that number is unverifiable
# here: this package's development hardware is an H100, so a test asserting the threshold is right
# would be asserting a number nobody measured on the machine running it. What IS testable -- which
# functional variants take the default, when the wide tile is admissible, that the arch fallback
# warns, and that the pool never empties -- is, and needs no GPU.
#
# The measured path (``do_autotune=True``) and the freeze are GPU tests and are marked as such.


#: Every test here is `@matrix_exempt` for ONE reason, so it is written once and referenced by
#: each: this module's subject is CONFIG SELECTION, not the kernel. Its cells are heuristic inputs
#: and pool members -- neither is a shape or dtype axis of the kernel, and the kernel itself is
#: exercised by ``test_layernorm_dual_gated_gemm.py``, which does draw from the matrix. The reason
#: must be a string LITERAL at each decoration site (the lock reads it from the AST), so it cannot
#: be hoisted into a constant.


def _autotune_operands(M=256, N=128, K=128, device="cuda"):
    """The five positional operands, sized so every pool member is admissible."""
    dt = torch.bfloat16
    return (
        torch.randn(M, K, device=device, dtype=dt),
        torch.randn(K, device=device, dtype=torch.float32),
        torch.randn(N, K, device=device, dtype=dt) * 0.1,
        torch.randn(N, K, device=device, dtype=dt) * 0.1,
        torch.empty(M, N, device=device, dtype=dt),
    )


# ───────────────────────────── the heuristic, GPU-free ─────────────────────────────


@pytest.mark.parametrize("flag", ["has_gate3", "has_mask", "transpose_out", "has_xgate"])
@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_every_functional_variant_takes_the_always_valid_default(flag):
    """Any functional flag returns ``{}`` -- the kernel's own defaults.

    This is the branch that makes the heuristic INERT for the TriMul workflow, whose front carries
    gate3 and a transposed store and whose back carries ``x_gate``. It is also what keeps the
    untested arch constant unreachable from the workflow: it is only consulted on the plain dual.

    ``has_xgate`` is included although nothing can set it yet. If a future variant were to default
    into the plain branch, it would be handed a ping-pong config it cannot run -- and a test written
    only for today's flags would not notice.
    """
    assert alg_fold_heuristic_config(4096, 128, 128, **{flag: True}) == {}


@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_a_small_contraction_picks_the_two_warpgroup_schedule():
    """Below the threshold the plain dual takes ping-pong, at a tile_M that admits it."""
    assert alg_fold_heuristic_config(4096, 128, 128) == {"pingpong": True}


@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_a_large_contraction_picks_the_wide_tile_when_it_divides():
    """At or above the threshold the wide cooperative tile wins -- if 2N can be tiled by it."""
    assert alg_fold_heuristic_config(4096, 512, 128) == {"tile_N": 256}


@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_a_pre_activation_the_wide_tile_cannot_divide_falls_back_to_the_default():
    """``2N % 256 != 0`` is a VALIDITY failure, not a preference, so the pick must not be 256.

    N=96 gives 2N=192, which no 256-wide tile divides. Returning ``{"tile_N": 256}`` here would
    hand the front door a config it refuses -- turning a heuristic into a crash, which is the one
    thing a no-timing pick must never do.
    """
    assert alg_fold_heuristic_config(4096, 512, 96) == {}


@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_the_threshold_is_read_from_the_arch_table_and_an_untuned_arch_warns(monkeypatch):
    """A non-tuned arch warns ONCE and still returns a runnable config.

    The warning is the whole mitigation for a ported-not-validated constant: without it the
    thresholds are used silently on hardware they were never bisected on, which is indistinguishable
    from using them on hardware they were.
    """
    for key in list(_WARNED):
        _WARNED.discard(key)
    monkeypatch.setenv("CPO_HEURISTIC_ARCH", "H100_SXM5")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cfg = alg_fold_heuristic_config(4096, 128, 128, device=torch.device("cpu"))
    assert cfg == {"pingpong": True}
    assert len(rec) == 1 and TUNED_ARCH in str(rec[0].message)


@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_omitting_the_device_resolves_the_tuned_arch_without_warning():
    """``device=None`` is the offline-audit path: the tuned arch's answer, and no noise.

    It is separated from the warning test because the two states are easy to conflate and the
    consequence of conflating them is a harness that records one arch's pick beside another arch's
    measurement -- silent on the tuned arch, wrong everywhere else.
    """
    for key in list(_WARNED):
        _WARNED.discard(key)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        alg_fold_heuristic_config(4096, 128, 128)
    assert len(rec) == 0


# ───────────────────────────── the pool, GPU-free ─────────────────────────────


@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_the_declared_space_is_the_product_of_its_axes():
    """Six points, and a FRESH object each call so a caller cannot mutate the decorated pool."""
    space = alg_fold_tuning_space()
    assert len(space.configs()) == 6
    assert space is not ALG_FOLD_TUNING_SPACE
    assert {tuple(sorted(c.all_kwargs())) for c in space.configs()} == {("pingpong", "tile_N")}


@pytest.mark.parametrize(
    "tile_N,pingpong,two_n,want",
    [
        (128, False, 256, True),
        (256, False, 256, True),
        (256, True, 256, False),  # ping-pong caps tile_N at 208
        (128, True, 256, True),
        (128, False, 192, False),  # 192 % 128 != 0
        (64, False, 192, True),
    ],
)
@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_a_config_survives_pruning_exactly_when_the_front_door_would_accept_it(
    tile_N, pingpong, two_n, want
):
    """The prune must agree with the kernel's own refusals, or the sweep times a crash.

    The ping-pong row is the one worth writing down: ``tile_N=256`` with ``pingpong=True`` is
    refused because the two-warpgroup schedule caps the CTA tile N at 208 for ``tile_M=128``. That
    is not deducible from the tile alone and was found by a perf-harvest run failing on it.
    """
    request = {
        "x": torch.empty(1024, 128),
        "Wg": torch.empty(two_n // 2, 128),
    }
    assert alg_fold_config_is_valid({"tile_N": tile_N, "pingpong": pingpong}, request) is want


@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_the_pool_can_never_empty_for_a_shape_the_front_door_accepts():
    """Some config must always survive, or the kernel is left with nothing to run.

    Swept over every 2N the front door admits at this floor. An empty pool is not a slow kernel; it
    is a `RuntimeError` from the tuner naming candidates rather than the shape that caused it.
    """
    for two_n in range(64, 1025, 64):
        request = {"x": torch.empty(1024, 128), "Wg": torch.empty(two_n // 2, 128)}
        survivors = [
            c for c in ALG_FOLD_TUNING_SPACE.configs() if alg_fold_config_is_valid(c, request)
        ]
        assert survivors, f"2N={two_n} pruned every config"


@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_the_k_tile_is_not_a_tuned_axis():
    """``blk_k`` decides how the k-loop associates its sums, so tuning it would tune the ANSWER.

    Two values give two different bit patterns for the same mathematical result. A pool containing
    it would make the numerical output depend on a timing measurement -- reproducible only for as
    long as the machine is idle in the same way.
    """
    knobs = {k for c in ALG_FOLD_TUNING_SPACE.configs() for k in c.all_kwargs()}
    assert "blk_k" not in knobs and "tile_M" not in knobs


# ───────────────────────────── the tuned entry, on device ─────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_the_fixed_path_accepts_the_knobs_the_tuned_path_refuses():
    """With the gate off the knobs are ordinary arguments -- that is how a heuristic pick is applied.

    Both halves matter: the fixed path MUST accept them (otherwise the heuristic's output cannot be
    used at all), and the tuned path must REFUSE them (otherwise the measured winner and the
    executed kernel are different kernels).
    """
    ops = _autotune_operands()
    layernorm_dual_gated_gemm_alg_fold(*ops, tile_N=128, pingpong=False)
    torch.cuda.synchronize()
    with pytest.raises(ValueError, match="tuned knobs"):
        layernorm_dual_gated_gemm_alg_fold(*ops, tile_N=128, do_autotune=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_the_heuristic_pick_is_runnable_at_every_shape_it_is_consulted_for():
    """The heuristic and the front door must agree; a pick the kernel refuses is worse than none.

    Runs each branch's output through the real entry rather than asserting the dict, because the
    dict being *correct* and the dict being *runnable* are different claims and only the second one
    is what a caller depends on.
    """
    for K, N in ((128, 128), (512, 128), (512, 96)):
        ops = _autotune_operands(M=256, N=N, K=K)
        cfg = alg_fold_heuristic_config(256, K, N, device=ops[0].device)
        layernorm_dual_gated_gemm_alg_fold(*ops, **cfg)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@matrix_exempt(
    "the subject is config SELECTION, not a kernel launch over shapes: these cells are heuristic inputs and pool members, and the kernel itself is covered by test_layernorm_dual_gated_gemm.py"
)
def test_a_swept_config_and_a_frozen_one_agree_bitwise():
    """Freezing must change WHICH config runs, never what it computes.

    Every pool member is a perf knob, so all of them produce the same answer; the frozen callable
    then has to reproduce that answer exactly. A difference here would mean a knob leaked into the
    arithmetic -- which is what keeping ``blk_k`` out of the pool is meant to prevent, asserted
    end-to-end rather than by inspecting the pool.
    """
    ops = _autotune_operands()
    reference = torch.empty_like(ops[4])
    layernorm_dual_gated_gemm_alg_fold(*ops[:4], reference, tile_N=128, pingpong=False)
    torch.cuda.synchronize()

    frozen = alg_fold_freeze(*ops)
    assert frozen.config is not None
    out = torch.empty_like(ops[4])
    frozen(*ops[:4], out)
    torch.cuda.synchronize()
    assert torch.equal(out, reference), f"frozen config {frozen.config} changed the result"


@requires_sm90
@pytest.mark.parametrize("D", [32, 64, 128, 256])
@matrix_exempt(
    "the subject is an OPERAND's storage extent, which is host-side bookkeeping; the widths are "
    "chosen so three of them need padding and one (D=256, where n3 == tile_N) does not, because a "
    "test that only saw padded cases could not tell the guard from an unconditional copy"
)
def test_gate3_vectors_are_padded_to_a_whole_tile(D, monkeypatch):
    """The output gate's row vectors own a whole tile of storage, so the epilogue cannot read past.

    Purpose
        A regression test for a MEMORY-SAFETY fault, not for a number. The epilogue's broadcast load
        in ``_internal/epi_ops.py`` reads the vector unconditionally and applies its predicate to the
        VALUE afterwards, so the lanes covering ``[n3, tile_N)`` still issue the global read. The
        dual region is safe because ``2n % tile_N == 0`` is enforced; the gate region has no such
        guarantee, and its weight rows are padded while its row vectors were not.

    Semantics
        Covers the CALL SITES, not just the helper. All three gate vectors -- the fold's ``colsum3``
        and ``dbias3`` and the gate bias ``b3`` -- are padded at separate places in the front door,
        and a helper that is correct but wired in at two of the three leaves the third reading off
        the end. So `_pad_gate3_vec_to_tile` is wrapped for the duration of one real gated launch
        and every call it receives is recorded; the assertions are on what actually reached it.

        Asserts the STORAGE, not the extent. The extent must stay ``n3`` -- it is baked into the
        compiled signature, so widening it would recompile -- and what changed is only that the
        bytes past it are inside the allocation and zero. ``D=256`` is in the pool as the case that
        needs no padding: it is what distinguishes a correct guard from a copy that always fires,
        and it was the one width that was clean before the fix.

    Args:
        D: The feature width, which is also the output gate's ``n3``.
        monkeypatch: pytest's fixture, used to install the recording wrapper for one test only.
    """
    import fold_cp_ops.kernels.layernorm_dual_gated_gemm as mod

    dev, M = "cuda", 256
    n = 2 * D
    tile_N = next(t for t in (256, 128, 64, 32, 16) if (2 * n) % t == 0)
    seen = []
    real = mod._pad_gate3_vec_to_tile

    def recording(t, tn):
        """Forward to the real pad, recording the tile it was asked for and what came back."""
        out = real(t, tn)
        seen.append((tn, None if t is None else tuple(t.shape), out))
        return out

    monkeypatch.setattr(mod, "_pad_gate3_vec_to_tile", recording)

    g = torch.Generator(device=dev)
    g.manual_seed(D)
    r = lambda *s, d=torch.bfloat16: torch.randn(*s, device=dev, dtype=d, generator=g)  # noqa: E731
    mod.layernorm_dual_gated_gemm(
        r(M, D),
        r(D, d=torch.float32),
        r(n, D) / D**0.5,
        r(n, D) / D**0.5,
        torch.empty(M, n, device=dev, dtype=torch.bfloat16),
        128,
        tile_N,
        norm_bias=r(D, d=torch.float32),
        W3=r(D, D) / D**0.5,
        b3=r(D, d=torch.float32),
        PostAct3=torch.empty(M, D, device=dev, dtype=torch.bfloat16),
        fusion_variant="alg_fold",
    )
    torch.cuda.synchronize()

    padded = [s for s in seen if s[1] is not None]
    assert len(padded) == 3, (
        f"expected the fold's colsum3 and dbias3 and the gate bias b3 to be padded, got "
        f"{len(padded)} call(s): {[s[1] for s in seen]} -- a dropped call site leaves that "
        f"vector's tail readable off the end"
    )
    want = (D + tile_N - 1) // tile_N * tile_N
    for tn, shape, out in padded:
        assert tn == tile_N, f"padded against tile {tn}, but the launch uses {tile_N}"
        assert out.shape[-1] == D, (
            f"the pad must keep the true extent {D}: it is baked into the compiled signature, "
            f"got {out.shape[-1]}"
        )
        have = out.untyped_storage().size() // out.element_size()
        assert have >= want, (
            f"a gate vector at D={D} owns {have} elements of storage but the epilogue reads "
            f"{want} (tile_N={tile_N}); the lanes past n3={D} would read off the end"
        )


@matrix_exempt(
    "the subject is WHICH variants wire the persistent-tile stats-race guard, a per-class "
    "structural fact; no shape or dtype varies and no kernel is launched"
)
def test_only_the_prolog_ln_variants_wire_the_stats_race_guard():
    """The leading stats barrier is passed by the `prolog_ln` classes and by no others.

    Purpose
        `finalize_row_stats` is a SHARED helper with five callers spanning two upstream lineages,
        and whether it should emit a leading ``arrive_and_wait()`` is a fact about the lineage, not
        about the helper. Upstream `staged` declares ``_stats_race_guard = True`` and applies it at
        both of its finalize entries; `stagec` and `layernorm_gemm_stagec` contain no such guard at
        all. So the `prolog_ln` callers need it and the `alg_fold` ones must not have it.

        **This exact distinction was already lost once.** An earlier revision removed the barrier
        from every caller and justified it with "the kernel this reproduces has no such barrier
        either" -- true of `stagec`, false of `staged`. A per-caller fact was applied globally, and
        nothing failed. This test is what fails.

    Semantics
        Structural, by AST, for two reasons. It asserts the property in BOTH directions -- present
        here, ABSENT there -- and an absence cannot be observed by running a kernel. And it cannot
        be made vacuous by a warm artifact cache: a runtime probe of this wiring only fires when a
        compile actually traces, so a cached kernel would let it pass having checked nothing, which
        is a failure mode this file has already been bitten by.

        Deliberately NOT a correctness test. The hazard is an intermittent write-after-read race
        across persistent tiles, so a green correctness run is not evidence the barrier is
        unnecessary and a red one would be luck. The property that was actually decided is which
        variants pass the flag, and that is what is pinned here.

    Raises:
        AssertionError: If a `prolog_ln` class stops passing it, if an `alg_fold` class starts, or
            if a new caller appears that this test does not know about -- the last one on purpose,
            so a sixth call site has to state its lineage rather than inherit a default silently.
    """
    from fold_cp_ops.kernels import layernorm_dual_gated_gemm as mod

    tree = ast.parse(inspect.getsource(mod))
    wiring = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for call in (n for n in ast.walk(cls) if isinstance(n, ast.Call)):
            if getattr(call.func, "id", None) != "finalize_row_stats":
                continue
            passed = any(
                kw.arg == "leading_barrier" and getattr(kw.value, "value", None) is True
                for kw in call.keywords
            )
            wiring.setdefault(cls.name, []).append(passed)

    want = {
        "LayerNormDualGatedGemmSm90": True,  # upstream `staged`      -- HAS the guard
        "LayerNormDualGatedGemmXGatePrologLnSm90": True,  # `staged` x_gate       -- HAS the guard
        "LayerNormDualGatedGemmAlgFoldSm90": False,  # upstream `stagec`     -- no guard
        "LayerNormDualGatedGemmXGateAlgFoldSm90": False,  # `stagec` x_gate       -- no guard
    }
    assert set(wiring) == set(want), (
        f"the set of classes calling finalize_row_stats changed: {sorted(wiring)} vs "
        f"{sorted(want)}. A new caller must declare which upstream lineage it reproduces -- "
        f"`staged` carries the guard, `stagec` does not -- rather than inherit a default."
    )
    for cls, calls in wiring.items():
        for got in calls:
            assert got is want[cls], (
                f"{cls} passes leading_barrier={got}, expected {want[cls]}. Adding it where the "
                f"upstream has none is adding synchronization rather than restoring it; dropping "
                f"it where the upstream has it reopens the persistent-tile stats race."
            )
