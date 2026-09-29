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
"""Tests for `fold_cp_ops.kernels.layernorm_gemm` -- the single-projection LayerNorm-fused GEMM.

**Two gates, and which one a property gets is a decision, not a default.** The reference comparison
is elementwise against an fp64 oracle with a DERIVED bound (`alg_fold`'s own, from
`fold_cp_ops.testing.numerics`), because the fusion never forms ``LayerNorm(x)`` and its error is
therefore a different function of the input from an unfused LayerNorm-then-GEMM's -- see
`numerics.alg_fold_error_bound` for why the two bounds are not interchangeable in either direction.
But several of this kernel's degrees of freedom are pure TRANSPORT: the CTA tile and the
cooperative/ping-pong schedule change which warp owns which output row and in what order the tiles
are visited, and must not change the arithmetic at all. For those the gate is `torch.equal`, because
a tolerance would absorb exactly the defect they exist to catch.

**The accumulation grouping is set by ``blk_k``, not by the tile.** That is what makes the bitwise
tile comparison legal: `resolve_blk_k` reads only the feature width, so every tile at one width sums
the same numbers in the same order.

**The oracle is fp64 and the bound is per element.** A scalar tolerance on a fused LayerNorm is
meaningless: the fold computes an off-centre row as the difference of two large nearly-equal
quantities, so the admissible error on one element can be orders of magnitude larger than on
another. `row_mean` is an axis for exactly that reason -- it is a property of the INPUT, not of the
kernel, and a zero-mean input hides the regime where this fusion is weakest.
"""

import math

import cutlass
import pytest
import torch

from fold_cp_ops._internal.autotune import AutotuneConfig
from fold_cp_ops.kernels.layernorm_gemm import (
    FUSION_VARIANTS,
    LayerNormGemmSm90,
    build_folded_operands,
    layernorm_gemm,
    layernorm_gemm_config_is_valid,
    layernorm_gemm_freeze,
    layernorm_gemm_heuristic_config,
    layernorm_gemm_tuning_space,
    resolve_blk_k,
)
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    dtype_facets,
    front_door_raises,
    int_facets,
    matrix_exempt,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt
from fold_cp_ops.testing.numerics import (
    _alg_fold_preact_error,
    _alg_fold_terms,
    _PRECISION_BITS,
    _U_FP32,
    assert_bitwise,
    assert_elementwise,
)

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(_SM != 9, reason="fold-cp-ops kernels require SM90 (H100/H200)")


# ── the declared test matrix for this kernel ───────────────────────────────────────────────────
# **Add a shape HERE, not to one test.** Every test below draws on this; narrowing it requires a
# written `because=`. Enforced at collection by tests/conftest.py; see
# fold_cp_ops/testing/kernel_matrix.py for the mechanics.
LAYERNORM_GEMM = KernelMatrix(
    kernel="layernorm_gemm",
    axes=(
        Axis(
            name="D",
            domain=(
                "the feature width, which is simultaneously the contraction extent K, the B row "
                "count and the output width P. Any positive int; the ONLY constraint is the "
                "16-byte alignment floor (D % 8 for a 16-bit activation). NOT restricted to a "
                "power of two, a tile multiple, or a multiple of 32 -- this kernel is the cp=1 "
                "workflow's universal fallback precisely for the widths the others refuse"
            ),
            values=(
                # the widths the A2A-fused TriMul workflow runs
                128,
                256,
                384,
                512,
                # 8-mod-16: NO allowed k-tile divides these, so the mainloop runs the 16-wide tile
                # with a TMA-zero-filled partial LAST k-tile. This is the case that distinguishes
                # this kernel's `resolve_blk_k` from the padding policy its sibling uses, and it is
                # where a mis-ported tiling shows up as a last-bit difference rather than a crash.
                136,
                200,
                264,
                520,
                # 16-aligned but not 64: exercises the 32- and 16-wide EXACT tiles
                192,
                320,
                # refused at the front door: not 16-byte aligned
                129,
                132,
            ),
            # tile=128 is the epilogue's N tile, so `tile_aligned` asks whether the trailing partial
            # N tile is predicated at all; big=512 is the top of the production range; small=128 its
            # bottom.
            facets={
                **int_facets(tile=128, big=512, small=128),
                # The four TriMul feature widths, under the name the workflow table uses.
                # Added ALONGSIDE the generic size facets rather than replacing them --
                # those describe tiling regimes, this one describes the production widths,
                # and the two overlap without meaning the same thing.
                "workflow_D": lambda v: v in (128, 256, 384, 512),
            },
        ),
        Axis(
            name="M",
            domain=(
                "the token count, entirely free -- no tile multiple, no power of two. It is a "
                "SYMBOLIC extent in the compiled kernel, so it never causes a recompile. The pool "
                "spans below one CTA tile, off-grid, and the MULTI-WAVE regime a persistent grid "
                "only reaches past ~132 CTAs' worth of tiles, which is where a per-CTA scratch "
                "reused across work tiles is exercised at all. The four largest are the A2A-fused "
                "TriMul token counts, N_token**2 / cp at cp=16 for N_token in "
                "2048/4096/8192/12288"
            ),
            values=(64, 199, 512, 4096, 262144, 1048576, 4194304, 9437184),
            facets={
                "partial_tile": lambda v: v < 128,
                "off_grid": lambda v: v % 128 != 0,
                "multi_wave": lambda v: v >= 262144,
                "production_scale": lambda v: v >= 4194304,
                # The declared FRONT series, under the name the table uses. Added ALONGSIDE
                # the facet above rather than replacing it: that one is guarding these values
                # today, and a rename would drop a live guard to gain a name. Two names for
                # one set is the cheap side of that trade.
                "workflow_M": lambda v: v in (262144, 1048576, 4194304, 9437184),
            },
        ),
        Axis(
            name="input_dtype",
            domain="float16 / bfloat16 -- the 16-bit operands the SM90 WGMMA atom takes here",
            values=(torch.bfloat16, torch.float16),
            facets=dtype_facets((torch.bfloat16, torch.float16)),
        ),
        Axis(
            name="has_bias",
            domain=(
                "whether a LayerNorm bias (beta) is given. Its ABSENCE removes the epilogue's `d` "
                "term entirely rather than adding a zero, so the two are different kernels"
            ),
            values=(False, True),
            facets={"with_bias": lambda v: v, "without_bias": lambda v: not v},
        ),
        Axis(
            name="has_gate3",
            domain=(
                "whether the trailing per-element output gate is fused. Absent, the kernel takes no "
                "C operand at all -- no TMA descriptor, no staging buffer, no epilogue multiply"
            ),
            values=(False, True),
            facets={"with_gate3": lambda v: v, "without_gate3": lambda v: not v},
        ),
        Axis(
            name="has_out_bias",
            domain=(
                "whether a bias on the GEMM's OUTPUT is given. Distinct from `has_bias`, which is "
                "the LayerNorm's beta on the normalized INPUT and is folded into a different vector"
            ),
            values=(False, True),
            facets={"with_out_bias": lambda v: v, "without_out_bias": lambda v: not v},
        ),
        Axis(
            name="fusion_variant",
            domain=(
                "which LayerNorm fusion the front door names, a member of FUSION_VARIANTS. "
                "Declared as an axis from day one even though one value is refused, so that adding "
                "the second is a NEW VALUE rather than a signature change -- and so the refusal is "
                "a tested fact rather than an absence"
            ),
            values=FUSION_VARIANTS,
            facets={
                "alg_fold": lambda v: v == "alg_fold",
                "prolog_ln": lambda v: v == "prolog_ln",
            },
        ),
        Axis(
            name="tile_M",
            domain=(
                "CTA tile M: 64/128/192/256/320 cooperatively, 64/128/192 under ping-pong. "
                "GemmSm90.__init__ raises on the rest. PURE TRANSPORT -- it decides which warp owns "
                "which output row, never how a row is summed"
            ),
            values=(64, 128, 192, 256),
            facets={
                "narrow": lambda v: v <= 64,
                "wide": lambda v: v >= 256,
                "pingpong_legal": lambda v: v in (64, 128, 192),
            },
        ),
        Axis(
            name="tile_N",
            domain=(
                "CTA tile N: divisible by 16 and <= 256, or by 32 and <= 512, with narrower limits "
                "under ping-pong. PURE TRANSPORT. It carries NO divisibility requirement on D -- "
                "the epilogue predicates a partial N tile, which is what lets 128 be the universal "
                "pick at every 16-byte-aligned width"
            ),
            values=(64, 128, 208, 256),
            facets={
                "narrow": lambda v: v <= 64,
                "wide": lambda v: v >= 256,
                "non_power_of_two": lambda v: v & (v - 1) != 0,
            },
        ),
        Axis(
            name="pingpong",
            domain=(
                "the two-warpgroup alternating schedule. It changes the statistics scratch from one "
                "block to two and the publishing barrier from the epilogue's to none at all, so it "
                "is the axis a scratch-lifetime bug appears on -- and it is still PURE TRANSPORT: "
                "the arithmetic per output tile is unchanged"
            ),
            values=(False, True),
            facets={"cooperative": lambda v: not v, "pingpong": lambda v: v},
        ),
        Axis(
            name="row_mean",
            domain=(
                "any float. NOT a kernel parameter -- it is a property of the INPUT, and it earns "
                "an axis because this fusion computes an off-centre row as the DIFFERENCE of two "
                "large nearly-equal quantities. Its conditioning is proportional to |mu|/sigma, so "
                "the regime where the algebraic fold is weakest is invisible at the zero mean "
                "torch.randn produces"
            ),
            values=(0.0, 10.0, 100.0),
            facets={
                "zero_mean": lambda v: v == 0.0,
                "off_centre": lambda v: v != 0.0,
                "badly_conditioned": lambda v: abs(v) >= 100.0,
            },
        ),
        Axis(
            name="arg_fault",
            domain=(
                "a malformed TENSOR ARGUMENT, or 'none'. The values name a CALLER MISTAKE rather "
                "than a kernel configuration, and every non-'none' one must be refused at this "
                "kernel's OWN front door with an API-level error. The pool spans the dtype AND the "
                "extent, and both an operand (B) and a non-operand (weight, gate3, out_bias), "
                "because which of the two a tensor is does not appear in the signature and so must "
                "not decide whether the mistake is named"
            ),
            values=(
                "none",
                "weight_dtype",
                "weight_extent",
                "B_extent",
                "gate3_extent",
                "gate3_dtype",
            ),
            facets={
                "well_formed": lambda v: v == "none",
                "bad_dtype": lambda v: v.endswith("_dtype"),
                "bad_extent": lambda v: v.endswith("_extent"),
                "non_operand": lambda v: v.startswith(("weight", "gate3")),
            },
        ),
    ),
    # Every region is PINNED on the axes `parametrize_unsupported` sweeps, so the emitted grid is
    # one cell per genuine refusal rather than a cross product of irrelevant combinations.
    # LayerNorm folded into the projection: a row reduction feeding a contraction, with an
    # optional sigmoid output gate.
    computes=("row_reduction", "contraction", "saturating_activation"),
    unsupported=(
        Unsupported(
            where=lambda D, input_dtype, fusion_variant, arg_fault: (
                fusion_variant == "prolog_ln"
                and D == 128
                and input_dtype == torch.bfloat16
                and arg_fault == "none"
            ),
            raises=ValueError,
            match=r"is not brought back",
            reason=(
                "prolog_ln -- the physical LayerNorm in the prologue -- is NOT brought back on this "
                "kernel: the cp=1 TriMul heuristic never returns it, so implementing it would add "
                "no reachable behaviour. The name is still declared, so adding it later is a new "
                "value rather than a signature change; this region is what keeps the refusal a "
                "tested fact instead of a TypeError from an unexpected keyword"
            ),
        ),
        Unsupported(
            where=lambda D, input_dtype, fusion_variant, arg_fault: (
                D % 8 != 0
                and input_dtype == torch.bfloat16
                and fusion_variant == "alg_fold"
                and arg_fault == "none"
            ),
            raises=ValueError,
            match=r"16-byte aligned",
            reason=(
                "the 16-byte TMA floor, and the ONLY shape constraint this kernel has. A width that "
                "is 4 mod 8 gives the A descriptor a row pitch the TMA cannot express, which faults "
                "in the driver rather than raising; an ODD width additionally makes the folded "
                "weight's row pitch unrepresentable. Both are refused at the front door in a "
                "sentence naming the width and the dtype that set the floor"
            ),
        ),
        Unsupported(
            where=lambda D, input_dtype, fusion_variant, arg_fault: (
                arg_fault == "weight_dtype"
                and D == 128
                and input_dtype == torch.bfloat16
                and fusion_variant == "alg_fold"
            ),
            raises=ValueError,
            match=r"weight must be",
            reason=(
                "the gain is folded into the weight in fp32 and rounded exactly ONCE; a 16-bit gain "
                "would round twice, losing more precision than the normalize it scales. Nothing "
                "downstream refuses it -- the fold kernel would happily read a bf16 vector as fp32 "
                "and produce plausible garbage -- so the check has to be here"
            ),
        ),
        Unsupported(
            where=lambda D, input_dtype, fusion_variant, arg_fault: (
                arg_fault == "weight_extent"
                and D == 128
                and input_dtype == torch.bfloat16
                and fusion_variant == "alg_fold"
            ),
            raises=ValueError,
            match=r"weight must be",
            reason=(
                "the gain is read with an unpredicated broadcast load, so one shorter than the "
                "feature width scales the tail of the fold with whatever follows it in memory -- a "
                "finite, plausible, entirely wrong weight, and no fault"
            ),
        ),
        Unsupported(
            where=lambda D, input_dtype, fusion_variant, arg_fault: (
                arg_fault == "B_extent"
                and D == 128
                and input_dtype == torch.bfloat16
                and fusion_variant == "alg_fold"
            ),
            raises=ValueError,
            match=r"B must be",
            reason=(
                "B's row count IS the contraction extent, and this kernel additionally requires its "
                "column count to equal it. A mismatched B is not a shape error anything downstream "
                "reports against an argument name -- it becomes a TMA descriptor over the wrong "
                "extent, reported against a symbol"
            ),
        ),
        Unsupported(
            where=lambda D, input_dtype, fusion_variant, arg_fault: (
                arg_fault == "gate3_dtype"
                and D == 128
                and input_dtype == torch.bfloat16
                and fusion_variant == "alg_fold"
            ),
            raises=ValueError,
            match=r"unsupported dtype for gate3",
            reason=(
                "gate3's EXTENT and its 16-byte row were both validated and its DTYPE was not, so "
                "an unsupported one reached `get_dtypes`' `torch2cute_dtype_map[...]` and surfaced "
                "as a bare `KeyError: torch.float64` -- naming a torch dtype, no argument, and not "
                "a type in API_LEVEL_ERRORS. `out_bias` three lines below has always had its dtype "
                "checked; gate3 was the sibling that did not. The front door's own docstring "
                "already promised a raise on a gate3 of the wrong shape OR dtype, so this makes "
                "the code match a contract it was already advertising"
            ),
        ),
        Unsupported(
            where=lambda D, input_dtype, fusion_variant, arg_fault: (
                arg_fault == "gate3_extent"
                and D == 128
                and input_dtype == torch.bfloat16
                and fusion_variant == "alg_fold"
            ),
            raises=ValueError,
            match=r"gate3 must be",
            reason=(
                "the gate rides the GEMM's C load, so its extents must match the output's exactly. "
                "A gate of the wrong shape is read through the output's descriptor, which reads "
                "past its end or silently drops its tail depending on which extent is wrong"
            ),
        ),
    ),
)


# ── oracle and bound ───────────────────────────────────────────────────────────────────────────


def reference_output(x, weight, bias, B, eps, gate3=None, out_bias=None):
    """The exact fp64 value the kernel approximates: ``(LayerNorm(x) @ B + out_bias) * gate3``.

    Purpose
        The oracle every correctness test compares against. It is written HERE rather than imported
        because the module it would come from on the upstream is deliberately not brought back -- it
        carries a second, unrelated kernel with it.

    Semantics
        Everything is promoted to fp64 FIRST and stays there, so the oracle carries no rounding of
        its own worth modelling: the LayerNorm, the contraction, the bias and the gate are each
        exact to within fp64. That is what lets the bound below describe the KERNEL's error alone.

    Args:
        x: ``(M, D)`` activation in the kernel's operand dtype. Not modified.
        weight: ``(D,)`` fp32 LayerNorm gain.
        bias: ``(D,)`` fp32 LayerNorm bias, or None. None is the no-beta LayerNorm, not a zero one --
            they agree numerically here but the kernel compiles differently for each.
        B: ``(D, P)`` weight in the operand dtype, with ``P == D``.
        eps: The variance floor. Must be the SAME value the kernel was given, or this describes a
            different function.
        gate3: ``(M, P)`` per-element gate, or None.
        out_bias: ``(P,)`` bias on the output, or None.

    Returns:
        An ``(M, P)`` fp64 tensor.
    """
    xn = torch.nn.functional.layer_norm(
        x.double(), (x.shape[-1],), weight.double(), None if bias is None else bias.double(), eps
    )
    out = xn @ B.double()
    if out_bias is not None:
        out = out + out_bias.double()
    if gate3 is not None:
        out = out * gate3.double()
    return out


def output_error_bound(x, weight, bias, B, reference, eps, out_dtype, gate3=None, out_bias=None):
    """Per-element allowance for ``(LayerNorm(x) @ B + out_bias) * gate3`` under the algebraic fold.

    Purpose
        The gate for every reference comparison here. A scalar tolerance cannot express this
        kernel's error: the fold computes an off-centre row as a cancellation, so the admissible
        deviation is a function of the row's conditioning and varies by orders of magnitude down one
        column.

    Semantics
        The pre-activation term is the SHARED per-projection quantity
        `numerics._alg_fold_preact_error` computes, reached through the same `_alg_fold_terms` the
        dual's bound uses -- so this bound and the dual's describe the same fusion rather than two
        readings of it. ``colsum_rounded=False`` because `build_folded_operands` reduces the fold
        BEFORE it is rounded into the weight dtype; passing True would attribute the error to the
        wrong term, which is unsound rather than merely loose.

        Three additions on top, in the epilogue's own order: the output bias is one fp32 add on the
        repaired accumulator; the gate is one fp32 multiply, which SCALES the pre-activation's error
        by the gate's magnitude; and the store rounds once to `out_dtype`.

    Args:
        x: ``(M, D)`` activation in the operand dtype.
        weight: ``(D,)`` fp32 gain. Passed at fp32 because the bound reproduces the STORED fold,
            which rounds it exactly once.
        bias: ``(D,)`` fp32 LayerNorm bias, or None. Must match what the kernel was given: its
            absence REMOVES a term rather than zeroing one.
        B: ``(D, P)`` weight in the operand dtype.
        reference: The exact fp64 ``(M, P)`` output, gate and biases included.
        eps: The variance floor the kernel was given.
        out_dtype: The dtype the output is stored as.
        gate3: ``(M, P)`` gate, or None.
        out_bias: ``(P,)`` output bias, or None.

    Returns:
        An fp64 tensor shaped like `reference`: the maximum absolute deviation each element may
        show.

    Raises:
        ValueError: From `_alg_fold_terms` -- if ``D * u >= 1``, or if a row is so nearly constant
            that the kernel's own fp32 ``rstd`` is undefined, which is an ill-conditioned input
            rather than a kernel defect.
    """
    terms = _alg_fold_terms(x, weight, bias, eps)
    # `_alg_fold_preact_error` wants the projection weight as (P, D) -- one row per OUTPUT column --
    # which is B transposed. Materialized rather than passed as a view because it is indexed as a
    # dense operand.
    err, mag = _alg_fold_preact_error(
        terms, weight, B.mT.contiguous(), out_bias, colsum_rounded=False
    )
    if out_bias is not None:
        err = err + _U_FP32 * mag
    if gate3 is not None:
        g = gate3.double().abs()
        err = err * g + _U_FP32 * (mag * g)
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    return err + half_ulp * (reference.abs() + err)


def make_inputs(M, D, dtype, has_bias, has_gate3, has_out_bias, row_mean=0.0, seed=0):
    """Build one well-formed argument set on the current CUDA device.

    Purpose
        One builder for every test, so a cell that fails cannot be a cell that was built
        differently. **Correctness-only, NOT performance-representative** -- it allocates the exact
        operands the front door wants and nothing else.

    Semantics
        `B` is scaled by ``1/sqrt(D)`` so the contraction's magnitude is independent of the feature
        width; without it the bound and the output both grow with ``D`` and a wide cell looks
        systematically worse than a narrow one for a reason that is not the kernel's.

        `row_mean` is added to `x` AFTER it is drawn, so the LayerNorm's own statistics see it. That
        is the point: it makes the fold's cancellation regime reachable, and it is applied before
        the cast so both dtypes see the same offset.

    Args:
        M: Token count. Any positive int.
        D: Feature width. Must satisfy the 16-byte floor for `dtype`, or the front door refuses it.
        dtype: The activation and weight dtype; fp16 or bf16.
        has_bias: Whether to build a LayerNorm bias.
        has_gate3: Whether to build the per-element output gate.
        has_out_bias: Whether to build the output bias.
        row_mean: Constant added to every element of `x`, shifting each row's mean.
        seed: Generator seed, so a failing cell is reproducible from its parameters alone.

    Returns:
        ``(x, weight, bias, B, gate3, out_bias)``, with the optional three None where not asked for.
    """
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    x = torch.randn(M, D, device=dev, dtype=torch.float32, generator=g)
    if row_mean:
        x = x + row_mean
    x = x.to(dtype)
    weight = torch.randn(D, device=dev, dtype=torch.float32, generator=g)
    bias = torch.randn(D, device=dev, dtype=torch.float32, generator=g) if has_bias else None
    B = torch.randn(D, D, device=dev, dtype=dtype, generator=g) / math.sqrt(D)
    gate3 = torch.randn(M, D, device=dev, dtype=dtype, generator=g) if has_gate3 else None
    out_bias = (
        torch.randn(D, device=dev, dtype=torch.float32, generator=g) if has_out_bias else None
    )
    return x, weight, bias, B, gate3, out_bias


def _check_against_reference(x, weight, bias, B, gate3, out_bias, eps, out, rows=None):
    """Compare `out` against the fp64 oracle under the derived bound, optionally on a slice of rows.

    Purpose
        The one comparison every correctness test funnels through, so the oracle, the bound and the
        assertion cannot drift apart between tests.

    Semantics
        `rows` exists for the production-scale token counts: an fp64 oracle over 9.4 million rows is
        several hundred gigabytes, so those cells check a leading SLICE. That is sound because the
        rows are independent -- the LayerNorm is per row and every output row is produced by one CTA
        from one row of `x` -- but it does NOT check the tail, which is why the off-grid and
        partial-tile cases are covered at small `M` where the whole output is checked.

    Args:
        x, weight, bias, B, gate3, out_bias: The arguments the kernel was given.
        eps: The variance floor it was given.
        out: The kernel's output.
        rows: How many leading rows to check, or None for all of them.

    Returns:
        The worst observed ``|err| / bound`` ratio, as a float.

    Raises:
        AssertionError: From `assert_elementwise`, if any checked element exceeds its bound.
    """
    if rows is not None and rows < x.shape[0]:
        x, out = x[:rows], out[:rows]
        gate3 = None if gate3 is None else gate3[:rows]
    ref = reference_output(x, weight, bias, B, eps, gate3, out_bias)
    bound = output_error_bound(
        x, weight, bias, B, ref, eps, out.dtype, gate3=gate3, out_bias=out_bias
    )
    return assert_elementwise(out, ref, bound, what="layernorm_gemm")


# ── correctness against the fp64 oracle ────────────────────────────────────────────────────────


@requires_sm90
@LAYERNORM_GEMM.parametrize(
    "D",
    "input_dtype",
    "has_bias",
    "M",
    only={"D": [128, 256, 384, 512], "M": [512]},
    because=(
        "the broad correctness grid, so D is the four widths the A2A-fused TriMul workflow actually "
        "runs and every one of them takes the 64-wide k-tile. The 8-mod-16 widths, the exact 32/16 "
        "tiles and the off-grid token counts have dedicated tests below, where they are the SUBJECT "
        "rather than a multiplier; carrying them here would cross them with 2 dtypes and 2 bias "
        "settings for no new signal at four cold compiles a cell. M is a symbolic extent so it "
        "never recompiles, and 512 checks the whole output against fp64 rather than a slice"
    ),
)
def test_matches_reference(D, input_dtype, has_bias, M):
    """The fused output matches an fp64 LayerNorm-then-GEMM oracle within the fold's own bound.

    This is the base correctness statement: at the four production feature widths, both 16-bit
    dtypes, and with and without the LayerNorm bias, every element lands inside the per-element
    allowance `output_error_bound` derives from the fusion's arithmetic.
    """
    eps = 1e-6
    x, w, b, B, _, _ = make_inputs(M, D, input_dtype, has_bias, False, False, seed=D + M)
    out = layernorm_gemm(x, w, B, bias=b, eps=eps, select="default")
    assert out.shape == (M, D) and out.dtype == input_dtype
    _check_against_reference(x, w, b, B, None, None, eps, out)


@requires_sm90
@LAYERNORM_GEMM.parametrize(
    "D",
    only={
        "D": [
            136,
            200,
            264,
            520,
            192,
            320,
        ]
    },
    because=(
        "the SUBJECT is the k-tiling, so this sweeps exactly the widths where it is not the plain "
        "64: 136/200/264/520 are 8 mod 16, which no allowed tile divides, so the mainloop runs the "
        "16-wide tile and the LAST k-tile is partial and TMA-zero-filled; 192 and 320 are 16- but "
        "not 64-aligned, taking the 64- and 32-wide EXACT tiles respectively. The other axes are "
        "pinned at bfloat16 with no bias so the cells differ only in the thing under test"
    ),
)
def test_off_grid_feature_dims(D):
    """Feature widths no allowed k-tile divides compute correctly, including the zero-filled tail.

    The partial last k-tile contributes zero to the contraction AND zero to the row statistics,
    while the LayerNorm normalizer stays the TRUE width -- so the padding is exact rather than
    merely out of range. A normalizer that used the padded extent instead would shrink every mean by
    a factor this test would catch, and a tail that was left undefined rather than zeroed would put
    a NaN or a stale value into a real product.
    """
    eps = 1e-6
    M = 512
    assert resolve_blk_k(D) in (64, 32, 16)
    x, w, b, B, _, _ = make_inputs(M, D, torch.bfloat16, False, False, False, seed=D)
    out = layernorm_gemm(x, w, B, eps=eps, select="default")
    _check_against_reference(x, w, b, B, None, None, eps, out)


@requires_sm90
@LAYERNORM_GEMM.parametrize(
    "has_gate3",
    "has_out_bias",
    "D",
    only={"D": [128, 264]},
    because=(
        "the SUBJECT is the two optional epilogue terms and their ORDER, which is independent of "
        "the feature width -- one width from each k-tiling regime is enough to show the terms are "
        "not tangled with the partial-tile path. The dtype is pinned at bfloat16, the noisier of "
        "the two, so a term applied in the wrong order cannot hide inside fp16's headroom"
    ),
)
def test_gate3_and_out_bias(has_gate3, has_out_bias, D):
    """The output bias lands OUTSIDE the rank-one repair and INSIDE the gate, in that order.

    ``out = (LayerNorm(x) @ B + out_bias) * gate3``. The order is not a preference: adding the bias
    before the ``r*`` scaling would bias every row by a per-row amount no reference computes, and
    applying the gate before the bias would gate the bias too. Both mistakes stay finite and
    plausible, which is why they are checked against an oracle that composes the terms explicitly
    rather than against a tolerance on the un-gated value.
    """
    eps = 1e-6
    M = 512
    x, w, b, B, g3, ob = make_inputs(
        M, D, torch.bfloat16, True, has_gate3, has_out_bias, seed=D + 7
    )
    out = layernorm_gemm(x, w, B, bias=b, eps=eps, gate3=g3, out_bias=ob, select="default")
    _check_against_reference(x, w, b, B, g3, ob, eps, out)


@requires_sm90
@LAYERNORM_GEMM.parametrize("row_mean", "input_dtype")
def test_off_centre_rows(row_mean, input_dtype):
    """An off-centre row is computed as a cancellation, and stays inside the bound that says so.

    The fold multiplies the RAW activation, so ``r*(x@Bw)`` and ``s*c`` are two large nearly-equal
    quantities whose difference is the answer. Every error in ``r``, in the accumulator and in ``c``
    is therefore scaled by the LARGE quantity and lands on the SMALL one, and the ratio between them
    is a property of the input. A zero-mean input -- which is all `torch.randn` produces -- never
    enters that regime at all, so without this test the bound's dominant term would be untested.
    """
    eps = 1e-6
    M, D = 512, 128
    x, w, b, B, _, _ = make_inputs(
        M, D, input_dtype, True, False, False, row_mean=row_mean, seed=11
    )
    out = layernorm_gemm(x, w, B, bias=b, eps=eps, select="default")
    _check_against_reference(x, w, b, B, None, None, eps, out)


@requires_sm90
@LAYERNORM_GEMM.parametrize(
    "M",
    "D",
    cells=[(M, D) for M in (64, 199, 4096) for D in (128, 200)],
    because=(
        "the SUBJECT is the token count at the two ends the tile does not divide: 64 is BELOW one "
        "CTA tile so the M predicate is engaged on the only wave, 199 is off-grid so the LAST wave "
        "is partial, and 4096 is a clean multiple that must stay unaffected. Crossed with one "
        "64-wide-k-tile width and one partial-k-tile width, because a partial M tile and a partial "
        "K tile are predicated by different machinery and their interaction is where a bound check "
        "written for one would be missing for the other. A `cells=` list rather than a product "
        "because the remaining M values are the production-scale ones, which have their own test"
    ),
)
def test_partial_and_off_grid_token_counts(M, D):
    """Token counts below and off the CTA tile compute every row, and no more than every row."""
    eps = 1e-6
    x, w, b, B, _, _ = make_inputs(M, D, torch.bfloat16, True, False, False, seed=M + D)
    out = layernorm_gemm(x, w, B, bias=b, eps=eps, select="default")
    assert out.shape == (M, D)
    _check_against_reference(x, w, b, B, None, None, eps, out)


@requires_sm90
@LAYERNORM_GEMM.parametrize(
    "M",
    "D",
    cells=[(262144, 128), (1048576, 256), (4194304, 384), (9437184, 512)],
    because=(
        "the A2A-fused TriMul token counts, N_token**2 / cp at cp=16 for N_token in "
        "2048/4096/8192/12288, each paired with the feature width that workflow runs it at. A "
        "`cells=` list rather than a product because the product is sixteen multi-gigabyte "
        "allocations to prove one property -- that a persistent grid reusing one per-CTA scratch "
        "across many work tiles does not carry a stale statistic into the next tile -- and the "
        "pairing already spans every width. The largest cell allocates over 20 GB and is skipped on "
        "OutOfMemoryError at RUNTIME, never by a memory estimate"
    ),
)
def test_trimul_token_counts(M, D):
    """The production token counts run, and the multi-wave persistent path keeps its rows straight.

    Past roughly 132 CTAs' worth of tiles a persistent grid walks many work tiles through ONE
    per-CTA statistics scratch, so tile *n+1*'s writes must not overtake tile *n*'s reads. What
    orders them is the epilogue's own rendezvous, which is why the reduction publishes through the
    epilogue barrier rather than a private one; a private barrier there would compile, would not
    hang, and would give the second tile the first tile's statistics on some rows. These are the
    shapes where that is reachable at all.

    Only the leading rows are checked against fp64: the oracle for the whole output would be
    hundreds of gigabytes. The rows are independent, so a slice is a sound check of the arithmetic;
    what it does not check is the tail, which the small-M tests cover exhaustively.
    """
    eps = 1e-6
    try:
        x, w, b, B, _, _ = make_inputs(M, D, torch.bfloat16, True, False, False, seed=D)
        out = layernorm_gemm(x, w, B, bias=b, eps=eps, select="default")
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip(f"M={M} D={D} does not fit in this device's memory")
    assert out.shape == (M, D)
    _check_against_reference(x, w, b, B, None, None, eps, out, rows=4096)
    del x, out
    torch.cuda.empty_cache()


# ── transport: the tile and the schedule must not move a single bit ────────────────────────────


@requires_sm90
@LAYERNORM_GEMM.parametrize(
    "tile_M",
    "tile_N",
    "pingpong",
    "D",
    cells=[(tm, 128, False, 128) for tm in (64, 128, 192, 256)]
    + [(128, tn, False, 128) for tn in (64, 208, 256)]
    + [(tm, 128, True, 128) for tm in (64, 128, 192)]
    + [(128, 128, True, 264), (256, 128, False, 264)],
    because=(
        "a `cells=` list because the tile axes are JOINT with the schedule axis -- ping-pong refuses "
        "tile_M above 192 outright and caps tile_N at 208 -- so the product would be mostly "
        "configurations the functor raises on, and a candidate that RAISES is not a slow schedule, "
        "it is a crashed sweep. The list spans every tile_M, every tile_N and both schedules, and "
        "repeats two of them at a partial-k-tile width because the statistics scratch is sized from "
        "tile_M while the k-tiling is sized from D, and their interaction is what a per-warpgroup "
        "scratch bug would hide behind"
    ),
)
def test_tile_and_schedule_are_bitwise_transport(tile_M, tile_N, pingpong, D):
    """Every tile and both schedules produce BITWISE the same output as the default configuration.

    The CTA tile decides which warp owns which output row and the schedule decides in what order the
    tiles are visited. Neither touches the arithmetic: the accumulation grouping is set by
    ``blk_k``, which `resolve_blk_k` derives from the feature width alone. So the gate is
    `torch.equal`, not a tolerance -- a tolerance here would absorb exactly the defects this test
    exists to catch, and there are two of them with no other detector.

    The first is a statistics scratch shared where it must not be: under ping-pong the two
    warpgroups own DIFFERENT output tiles concurrently, so one block per warpgroup is allocated and
    the publishing barrier must NOT be the epilogue's -- which is sized for one warpgroup under that
    schedule, so reusing it compiles, does not hang, and rendezvouses two warpgroups that are in
    different stages on different tiles. The second is a row-ownership mismatch between the
    reduction's partition and the epilogue's broadcast, which scales a row by another row's
    statistic. Both stay finite and plausibly scaled.
    """
    eps = 1e-6
    M = 512
    x, w, b, B, g3, ob = make_inputs(M, D, torch.bfloat16, True, True, True, seed=D + 3)
    baseline = layernorm_gemm(
        x, w, B, bias=b, eps=eps, gate3=g3, out_bias=ob, tile_M=128, tile_N=128, select="default"
    )
    out = layernorm_gemm(
        x,
        w,
        B,
        bias=b,
        eps=eps,
        gate3=g3,
        out_bias=ob,
        tile_M=tile_M,
        tile_N=tile_N,
        pingpong=pingpong,
        select="default",
    )
    assert_bitwise(
        out,
        baseline,
        what=f"tile=({tile_M}, {tile_N}) pingpong={pingpong} at D={D} moved {int((out != baseline).sum())} of {out.numel()} elements against the (128, 128) cooperative baseline. The tile and the schedule are transport: they must not change the arithmetic at all.",
    )


@requires_sm90
@LAYERNORM_GEMM.parametrize(
    "D",
    only={"D": [128, 200]},
    because=(
        "the SUBJECT is the activation's LAYOUT, which is a stride fact independent of everything "
        "else; one width from each k-tiling regime shows the transposing load is not entangled with "
        "the partial-tile path"
    ),
)
def test_layout_left_activation_is_bitwise_identical(D):
    """A layout-left activation gives BITWISE the same answer as its row-major copy.

    ``x`` with stride ``(1, M)`` compiles to its own kernel: `get_majors` reads the strides, the
    WGMMA takes an MN-major A operand, and the transpose that would otherwise need a separate pass
    disappears. The row statistics index the staged tile LOGICALLY, so the two artifacts must agree
    to the bit -- a difference would mean the reduction's partition had followed the physical layout
    instead, which is a wrong answer with no diagnostic.
    """
    eps = 1e-6
    M = 512
    x, w, b, B, _, _ = make_inputs(M, D, torch.bfloat16, True, False, False, seed=D + 5)
    x_left = x.T.contiguous().T
    assert x_left.stride() == (1, M) and torch.equal(x_left, x)
    row_major = layernorm_gemm(x, w, B, bias=b, eps=eps, select="default")
    left = layernorm_gemm(x_left, w, B, bias=b, eps=eps, select="default")
    assert_bitwise(
        left,
        row_major,
        what=f"the layout-left activation moved {int((left != row_major).sum())} elements at D={D}",
    )


# ── the refusals ───────────────────────────────────────────────────────────────────────────────


@requires_sm90
@LAYERNORM_GEMM.parametrize_unsupported("D", "input_dtype", "fusion_variant", "arg_fault")
def test_unsupported_raises(
    D, input_dtype, fusion_variant, arg_fault, expected_error, expected_match
):
    """Every declared unsupported combination is REFUSED at this kernel's own front door.

    Not `xfail`: an xfail'd correctness assertion cannot tell "refused" from "answered wrongly", so
    a kernel that silently computed garbage here would satisfy it and the suite would stay green.
    `pytest.raises` fails that kernel with "DID NOT RAISE".

    Nor an `AssertionError`: ``python -O`` strips asserts, so a guard written that way is not a check
    at all under optimization, and the internal explosion it was standing in front of comes back.
    Every one of these raises `ValueError` from the front door with a message naming the argument.
    """
    M = 512
    # Every tensor is built at THIS D, including the misaligned ones -- `make_inputs` only
    # allocates. That matters: the front door checks `weight` and `B` against `N` BEFORE it checks
    # the alignment, so a set built at a different width would trip the shape guard first and this
    # test would assert the wrong refusal while still passing.
    x, w, b, B, g3, ob = make_inputs(M, D, input_dtype, False, True, False)
    if arg_fault == "weight_dtype":
        w = w.to(input_dtype)
    elif arg_fault == "weight_extent":
        w = w[:-8].contiguous()
    elif arg_fault == "B_extent":
        B = B[:-8].contiguous()
    elif arg_fault == "gate3_extent":
        g3 = g3[:, :-8].contiguous()
    elif arg_fault == "gate3_dtype":
        g3 = g3.double()
    with front_door_raises(expected_error, expected_match):
        layernorm_gemm(
            x,
            w,
            B,
            bias=b,
            eps=1e-6,
            gate3=g3,
            out_bias=ob,
            fusion_variant=fusion_variant,
            select="default",
        )


@requires_sm90
@matrix_exempt("the subject is the SELECTION front door's own refusal, which no axis names")
def test_unknown_select_raises():
    """An unrecognised ``select`` is refused rather than silently falling through to the fixed path.

    The three modes differ in whether the caller's tile is honoured, swept over, or replaced, so a
    typo that fell through to one of them would run a configuration the caller did not ask for and
    report nothing.
    """
    x, w, _, B, _, _ = make_inputs(128, 128, torch.bfloat16, False, False, False)
    with pytest.raises(ValueError, match=r"select must be"):
        layernorm_gemm(x, w, B, select="hueristic")


@matrix_exempt("the subject is the FUNCTOR's constructor guards, which run with no device at all")
@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(fusion_variant="prolog_ln"), r"is not brought back"),
        (dict(fusion_variant="physical_ln"), r"fusion_variant must be one of"),
        (dict(blk_k=48), r"blk_k must be one of"),
        (dict(gemm_k=0), r"gemm_k must be"),
    ],
)
def test_functor_refuses_bad_parameters(kwargs, match):
    """The functor refuses an unbuildable parameter at construction, not three frames into a trace.

    The front door checks the variant first, so that one is a second line of defence -- but it is
    the line that holds for a caller building the functor directly, which the compile path does.
    ``blk_k`` and ``gemm_k`` have no front-door equivalent at all: they are derived rather than
    passed, so this is their only guard. A ``blk_k`` outside the allowed set would build a WGMMA
    K-block count the atom cannot feed, and a zero ``gemm_k`` would divide every row mean by zero.
    """
    base = dict(fusion_variant="alg_fold", blk_k=64, gemm_k=128)
    with pytest.raises(ValueError, match=match):
        LayerNormGemmSm90(
            cutlass.Float32, cutlass.BFloat16, (128, 128), (1, 1, 1), **{**base, **kwargs}
        )


# ── the host fold ──────────────────────────────────────────────────────────────────────────────


@requires_sm90
@LAYERNORM_GEMM.parametrize(
    "D",
    "input_dtype",
    "has_bias",
    only={"D": [128, 136, 384]},
    because=(
        "the SUBJECT is the host fold, which is a pure elementwise pass plus two column reductions "
        "and knows nothing about the mainloop's tiling -- so one 64-wide-k-tile width, one "
        "8-mod-16 width and one larger width span everything the kernel's grid does. Both dtypes "
        "because the fold ROUNDS into the weight dtype, and both bias settings because the bias "
        "reduction is compiled out entirely when there is none"
    ),
)
def test_fold_matches_torch(D, input_dtype, has_bias):
    """`build_folded_operands` reproduces the torch fold exactly where it can, and closely where not.

    The folded weight is BITWISE identical to multiplying and casting in torch, because it is pure
    elementwise work and there is nothing for a fusion to reorder. The two column sums are
    REDUCTIONS, so they agree only up to summation order: the kernel fixes its own order -- ascending
    within a row group, then row-group 0 folding the partials ascending -- so its output is
    reproducible run to run, but that order is not torch's. Using `torch.equal` for all three would
    fail for a reason that has nothing to do with the kernel.

    Also asserts the reproducibility itself, because a fixed order is the property that makes a
    later byte-identity comparison meaningful at all, and an atomic-based reduction would satisfy
    the tolerance while destroying it.
    """
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(D)
    B = torch.randn(D, D, device=dev, dtype=input_dtype, generator=g)
    w = torch.randn(D, device=dev, dtype=torch.float32, generator=g)
    b = torch.randn(D, device=dev, dtype=torch.float32, generator=g) if has_bias else None
    Bw, c, d = build_folded_operands(B, w, b)
    assert torch.equal(Bw, (w.unsqueeze(1) * B.float()).to(input_dtype)), (
        "the folded weight is pure elementwise work and must be bitwise identical to torch's"
    )
    c_ref = (w.unsqueeze(1) * B.float()).sum(0)
    # The reduction is fp32 over D terms in a different order from torch's, so the allowance is the
    # standard summation-growth factor against the sum of magnitudes -- not a fixed epsilon, which
    # would be wrong by orders of magnitude at one end of the width pool or the other.
    gamma = (D * _U_FP32) / (1.0 - D * _U_FP32)
    c_bound = gamma * (w.unsqueeze(1) * B.float()).abs().sum(0).double() + 1e-30
    assert_elementwise(c, c_ref, c_bound, what="colsum")
    if has_bias:
        d_ref = (b.unsqueeze(1) * B.float()).sum(0)
        d_bound = gamma * (b.unsqueeze(1) * B.float()).abs().sum(0).double() + 1e-30
        assert_elementwise(d, d_ref, d_bound, what="dbias")
    else:
        assert d is None, "no LayerNorm bias must compile the d term out, not produce a zero vector"
    Bw2, c2, d2 = build_folded_operands(B, w, b)
    assert torch.equal(c, c2) and torch.equal(Bw, Bw2), (
        "the fold's reductions run in a fixed order and must be bit-reproducible run to run; an "
        "atomic-based reduction would pass the bound above and fail this"
    )


# ── selection: the size heuristic and the declared sweep ───────────────────────────────────────


@LAYERNORM_GEMM.parametrize(
    "M",
    "D",
    cells=[(m, d) for m in (64, 512, 262144, 1048576) for d in (128, 256, 512, 200)],
    because=(
        "the SUBJECT is the heuristic's arithmetic, which reads only (M, D) and runs on the host, so "
        "the cells are chosen to straddle its two thresholds: the bulk/small-tile split at D == 128 "
        "and the ping-pong floor in M. 200 is there because an off-grid width between the "
        "thresholds must follow the BULK rule and not fall to the default -- a `>= 256` gate would "
        "wrongly drop it, and that is a bug this test would catch and a product over the whole D "
        "pool would bury"
    ),
)
@numeric_exempt("asserts the heuristic's pick is RUNNABLE at that shape; it launches nothing")
def test_heuristic_picks_a_runnable_config(M, D):
    """The size heuristic returns a configuration that CAN run the shape it was asked about.

    A ping-pong pick at a CTA-M the schedule refuses raises from the functor's constructor, so a
    heuristic that ignored that would turn a default-path call into a crash. A tile WIDER than the
    token count is a different matter -- it runs, wasting a wave -- and the heuristic is checked
    against it here as a performance property, with the always-valid ``(128, 128)`` fallback
    exempted because it is the tile every shape below one tile falls back to.

    This asserts validity, not optimality: the thresholds are a performance claim measured
    elsewhere, and pinning them here would make a re-tune look like a test failure.
    """
    cfg = layernorm_gemm_heuristic_config(M, D)
    kw = cfg.all_kwargs()
    assert kw["tile_M"] <= max(M, 128), "a tile wider than the token count wastes a whole M wave"
    if kw["pingpong"]:
        assert kw["tile_M"] in (64, 128, 192), "ping-pong refuses a wider CTA-M at construction"
    assert kw["tile_N"] == 128, (
        "tile_N=128 is the universal pick: the epilogue predicates a partial N tile, so it carries "
        "no divisibility requirement on D"
    )
    assert layernorm_gemm_config_is_valid(kw, {"x": torch.empty(M, D, device="meta")}), (
        "the heuristic must not return a configuration its own validity filter rejects"
    )


@matrix_exempt("the subject is the declared SWEEP SPACE itself, which no shape axis parametrizes")
def test_tuning_space_excludes_only_what_the_functor_refuses():
    """Every point in the declared sweep is a tile geometry `GemmSm90` will actually build.

    A candidate that RAISES is not a slow configuration, it is a crashed sweep -- and it crashes
    inside the tuner, where the failure is reported against a config index rather than a shape. The
    space therefore declares the full grid and EXCLUDES from it, so a combination nobody considered
    stays distinguishable from one considered and rejected.
    """
    space = layernorm_gemm_tuning_space()
    points = [c.all_kwargs() for c in space.configs()]
    assert points, "an empty sweep space would silently disable tuning"
    for p in points:
        if p["pingpong"]:
            assert p["tile_M"] in (64, 128, 192), p
            assert p["tile_N"] <= 208, p
    assert any(p["pingpong"] for p in points) and any(not p["pingpong"] for p in points), (
        "both schedules must be reachable, or the sweep cannot find the one the heuristic picks"
    )


@requires_sm90
@matrix_exempt("the subject is COMPILE TIME, which is a property of the artifact, not of a shape")
def test_cold_compile_under_the_bar():
    """A cold compile of the default configuration stays under the repository's five-second bar.

    Measured in a fresh interpreter with the disk cache disabled, at the width whose k-loop is
    LONGEST -- the 16-wide tile at 520 features is 33 unrolled k-tiles, where a nested branch inside
    an unrolled loop would blow up combinatorially. That is the known root-cause class, and the fix
    for a breach is code, never a cache.
    """
    import subprocess
    import sys

    script = (
        "import time, torch;"
        "from fold_cp_ops.kernels.layernorm_gemm import layernorm_gemm;"
        "x=torch.randn(512,520,device='cuda',dtype=torch.bfloat16);"
        "w=torch.randn(520,device='cuda',dtype=torch.float32);"
        "B=torch.randn(520,520,device='cuda',dtype=torch.bfloat16);"
        "t=time.perf_counter();"
        "layernorm_gemm(x,w,B,select='default');"
        "print(time.perf_counter()-t)"
    )
    env = {"CPO_CACHE_ENABLED": "0"}
    import os

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, **env},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    elapsed = float(proc.stdout.strip().splitlines()[-1])
    assert elapsed <= 5.0, (
        f"cold compile took {elapsed:.2f}s against the 5s bar. Look FIRST for a nested `if` inside "
        f"a `cutlass.range_constexpr` loop -- the fix is code, not a cache."
    )


@requires_sm90
@matrix_exempt(
    "the subject is the FREEZE entry point -- that it resolves a config and binds it -- which no "
    "shape or dtype axis varies"
)
def test_freezing_binds_a_config_and_the_frozen_call_reproduces_it():
    """`layernorm_gemm_freeze` resolves a pick, exposes it, and the bound callable reruns exactly it.

    **This function shipped broken and nothing noticed, which is why it is tested here rather than
    trusted.** It read ``.best_config`` off the ``@autotune``-decorated wrapper, which carries only
    ``.autotuner`` and ``.axes``, so the first call raised `AttributeError` -- after passing a full
    correctness suite and a 190-cell byte-identity sweep. Freezing runs a tuning sweep, so it is
    expensive to call, so nothing called it, so the accessor was never executed. The fix is one
    attribute; this test is what makes the fix mean something.

    Three assertions, and the third is the one a smoke test would miss:

    * the pick is exposed on ``.config`` and is a real `AutotuneConfig` -- **not None**. None is the
      quiet failure mode here: if the tuning gate ever stops being set, ``best_config`` stays None,
      `frozen_call` passes ``_config=None``, the tuner treats that as "no pin" and re-selects, and
      the frozen callable silently stops being frozen while still returning correct numbers;
    * the frozen output matches the fp64 oracle under the shared bound;
    * the frozen output is BITWISE equal to the same config dispatched explicitly, which is what
      says the callable ran its OWN pick rather than re-selecting.

    ``CPO_AUTOTUNE=0`` makes this one compile instead of a sweep, and does not weaken it: the tuner
    then takes the first admissible config rather than measuring, but still assigns
    ``best_config``, so the accessor that broke runs identically.
    """
    monkey = pytest.MonkeyPatch()
    monkey.setenv("CPO_AUTOTUNE", "0")
    try:
        M, D, P = 256, 128, 128
        torch.manual_seed(0)
        x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(D, device="cuda", dtype=torch.float32)
        bias = torch.randn(D, device="cuda", dtype=torch.float32)
        B = torch.randn(P, D, device="cuda", dtype=torch.bfloat16)
        frozen = layernorm_gemm_freeze(x, weight, B, bias=bias, eps=1e-6)
        assert isinstance(frozen.config, AutotuneConfig), (
            f"freeze must expose a real config on `.config`; got {frozen.config!r}. None means the "
            f"tuner never ran, and the 'frozen' callable would silently re-select on every call."
        )
        got = frozen(x, weight, B)
        _check_against_reference(x, weight, bias, B, None, None, 1e-6, got)
        pinned = layernorm_gemm(x, weight, B, bias=bias, eps=1e-6, _config=frozen.config)
        assert torch.equal(got, pinned), (
            f"the frozen callable did not run its own pick {frozen.config}: its output differs "
            f"from the same config dispatched explicitly"
        )
    finally:
        monkey.undo()


#: What is left to build for the register-source path, in order. Carried on both `xfail`
#: markers so `-k register_source` prints the whole map without opening this file.
#:
#: **Both markers pin `raises=`, and that is not decoration.** This file's own convention (see
#: `test_unsupported_combos_raise`) refuses a bare xfail on a correctness assertion, because such a
#: marker cannot tell "the kernel REFUSED" from "the kernel ANSWERED WRONGLY" -- a silently-wrong
#: kernel satisfies it and the suite stays green. A bare `xfail(strict=True)` here would have
#: exactly that hole: it stays green whether the knob is unreachable (today: `TypeError`) or the rs
#: branch is implemented and computing garbage (later: `AssertionError`), which are precisely the
#: two states that must not be conflated. Naming the type closes it -- the marker absorbs ONLY
#: "the knob is not wired", and a failure on NUMBERS is an unexpected exception type that the suite
#: reports. `strict=True` closes the other end: when the branch is correct the test passes, the
#: unexpected pass fails the suite, and the marker gets removed. The reminder cannot be forgotten
#: because it is enforced from both sides.
_RS_TODO = (
    "the register-source path is not wired yet: (1) thread a_in_regs/rs_ring into "
    "_compile_layernorm_gemm so they reach the artifact key; (2) expose them on layernorm_gemm(); "
    "(3) then the rs branch in mma_setup_fragments. The five forked methods and the front-door "
    "gate landed inert in c581339. raises= pins the failure MODE, so this marker absorbs only "
    "'the knob is unreachable' and never a wrong ANSWER; strict=True turns the eventual pass into "
    "a suite failure that says remove me."
)


@pytest.mark.xfail(raises=AssertionError, strict=True, reason=_RS_TODO)
@matrix_exempt(
    "the subject is whether the a_in_regs KNOB reaches the compiled artifact's key; no shape or "
    "dtype varies and nothing is launched"
)
def test_the_register_source_knobs_reach_the_compile_key():
    """`a_in_regs` and `rs_ring` must be arguments of the compile entry, and reachable from the API.

    Purpose
        **This is the FIRST test a new config knob gets, before any test of what the knob does.**
        A knob that does not reach the artifact key produces a silently wrong artifact: two configs
        differing only in that knob hash to the same key, and `jit_cache` is persistent and
        CROSS-PROCESS, so one silently serves the other. No numeric test can find that, because the
        numbers are correct for the artifact that actually ran -- it is simply the wrong artifact.

    Semantics
        Two assertions, and the order is the point:

        * the flags are parameters of `_compile_layernorm_gemm`, whose arguments ARE the disk-cache
          key (`jit_cache` builds it from ``(qualname,) + args``);
        * the flags are reachable from the public entry, because a knob that only the class
          constructor accepts cannot be exercised by any caller -- including the A/B test below,
          which would then compare two identical runs and report success.

        Structural rather than behavioural on purpose: it holds with no GPU, and it states the
        property (`the knob is keyed`) rather than a symptom of its absence.

    Raises:
        AssertionError: If either flag is missing from the compile key or from the public entry.
    """
    import inspect

    from fold_cp_ops.kernels.layernorm_gemm import _compile_layernorm_gemm, layernorm_gemm

    compile_fn = getattr(_compile_layernorm_gemm, "__wrapped__", _compile_layernorm_gemm)
    keyed = set(inspect.signature(compile_fn).parameters)
    missing_key = {"a_in_regs", "rs_ring"} - keyed
    assert not missing_key, (
        f"{sorted(missing_key)} are not arguments of _compile_layernorm_gemm, so they are NOT in "
        f"the artifact key. Two configs differing only in these would share one compiled .o "
        f"through the persistent cross-process cache -- a wrong-artifact bug, not a crash. "
        f"Keyed today: {sorted(keyed)}"
    )

    entry = set(inspect.signature(layernorm_gemm).parameters)
    missing_api = {"a_in_regs", "rs_ring"} - entry
    assert not missing_api, (
        f"{sorted(missing_api)} are not parameters of layernorm_gemm(), so no caller can select "
        f"the register-source path and the A/B comparison below would run the SAME path twice and "
        f"pass vacuously."
    )


@pytest.mark.xfail(raises=TypeError, strict=True, reason=_RS_TODO)
@requires_sm90
@matrix_exempt(
    "the subject is that two SCHEDULES compute one mathematical object; the tile is pinned to "
    "(128, 128) because that is the only one a_in_regs is valid at"
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_the_register_source_path_computes_what_the_shared_memory_path_does(has_bias):
    """``a_in_regs`` is a schedule choice, so both arms must produce the same numbers.

    Purpose
        The register-source path reduces the LayerNorm statistics out of the WGMMA operand-A
        fragment instead of re-reading the staged tile. That reduction depends on an EMPIRICALLY
        derived map from a thread to the two rows it owns, so the failure mode of getting it wrong
        is not a fault -- it is a plausible-looking LayerNorm computed with another row's mean.

        The shared-memory arm is the oracle: it is the shipping path, already under test at these
        shapes, and it needs no reference implementation of its own. Any mis-partition of ``tCrA``
        either faults or disagrees with it.

    Semantics
        Same operands, same tile, one knob apart. Compared to fp-ordering tolerance rather than
        bitwise: the two arms sum a row's K in different orders -- the register arm completes it
        across four quad lanes, the shared-memory arm across a butterfly -- so the last bits differ
        by construction, exactly as the two `finalize` paths' docstrings state.

        **THREE assertions, and the extra one is not redundant -- do NOT simplify this back to a
        bare `assert_close(regs, smem)`.** A pure A/B cannot see the two arms being wrong TOGETHER,
        which is the likely shape of the bug here: both arms read the same staged tile through the
        same fold, so a fold-level or launch-level mistake moves both and the A/B still passes. So
        each arm is first held INDEPENDENTLY against this file's fp64 oracle
        (`_check_against_reference`), and only then are the two required to agree with each other
        MORE TIGHTLY than either agrees with fp64. That ordering also buys a diagnosable failure:
        a mis-partitioned ``tCrA`` fails its own oracle check first, naming the arm, instead of
        surfacing as a symmetric two-arm disagreement that says nothing about which one is wrong.
        And the tight bound is DERIVED (the fold's own `_U_FP32 * 2**_PRECISION_BITS`), not an
        invented tolerance chosen until it passed.

    Raises:
        AssertionError: If either arm disagrees with fp64 beyond the fold's error bound, or if the
            two arms disagree by more than the K-summation-order effect. **This is NOT the type the
            `xfail` marker absorbs** -- it pins `TypeError`, the unreachable knob -- so a numeric
            disagreement is reported as a real failure rather than swallowed. See `_RS_TODO`.
        TypeError: Today, from `layernorm_gemm()` not accepting `a_in_regs`. This is the expected
            failure until the wiring lands.
    """
    D, M, eps = 128, 4096, 1e-6
    x, w, b, B, _, _ = make_inputs(M, D, torch.bfloat16, has_bias, False, False, seed=7)
    pin = dict(bias=b, eps=eps, select="default", tile_M=128, tile_N=128)
    smem = layernorm_gemm(x, w, B, **pin)
    regs = layernorm_gemm(x, w, B, a_in_regs=True, **pin)

    # BOTH arms against the SAME fp64 oracle, rather than against each other. Comparing the two
    # arms alone would need an invented tolerance and would pass if they were wrong TOGETHER;
    # holding each to the fold's own derived bound is the check the rest of this file uses.
    _check_against_reference(x, w, b, B, None, None, eps, smem)
    _check_against_reference(x, w, b, B, None, None, eps, regs)

    # And they must agree with each other far more closely than either agrees with fp64: they
    # differ only in the ORDER a row's K is summed (four quad lanes versus a butterfly), so a
    # disagreement larger than that order effect means the register arm read the wrong rows.
    assert_elementwise(
        regs.float(),
        smem.float(),
        _U_FP32 * (2**_PRECISION_BITS) * regs.float().abs().clamp_min(1.0),
        what="a_in_regs vs shared-memory schedule",
    )
