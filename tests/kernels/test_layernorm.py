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
# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# tests/test_layernorm.py
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

import pytest
import torch

import cutlass

from fold_cp_ops.kernels.layernorm import (
    _DELAY_W_EST_REG_THRESHOLD,
    _auto_subt,
    _subt_is_tileable,
    _transpose_default_config,
    layernorm_fwd,
    layernorm_ref,
    layernorm_rstd_ref,
    layernorm_mean_ref,
    layernorm_transpose_freeze,
    _compile_layernorm_fwd,
)
import fold_cp_ops.kernels.layernorm as _ln
import fold_cp_ops._internal.cache_utils as _cache_utils
from fold_cp_ops.testing.numerics import assert_bitwise, assert_elementwise, tolerance_bound
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    dtype_facets,
    front_door_raises,
    int_facets,
    matrix_exempt,
)


def _layout_left(base: torch.Tensor) -> torch.Tensor:
    """Copy ``base`` into a LayoutLeft ``(M, N)`` tensor, i.e. stride ``(1, M)``.

    The ``transpose=True`` variant reduces over N, the STRIDED axis, and refuses a row-major input
    at the front door. A plain ``torch.randn(M, N)`` is row-major, so every transposing test has to
    go through this -- building it by hand is the mistake the helper exists to remove.

    Args:
        base: Any 2-D CUDA tensor. Its values and dtype are preserved exactly (the copy is a layout
            change, not a cast), so a reference computed on ``base`` is the right oracle.

    Returns:
        A new tensor with the same shape, dtype and values, and stride ``(1, M)``.

    Raises:
        AssertionError: If the constructed stride is not ``(1, M)`` -- a torch change that broke the
            layout would otherwise turn every transposing test into a front-door refusal.
    """
    M, N = base.shape
    x = torch.empty_strided((M, N), (1, M), device=base.device, dtype=base.dtype).copy_(base)
    assert x.stride() == (1, M), x.stride()
    return x


# ── the declared test matrix for this kernel ───────────────────────────────────────────────────
# **Add a shape HERE, not to one test.** Every test below draws on this; narrowing it requires a
# written `because=`. tests/perf/test_benchmark_perf_layernorm.py imports this very object, so the
# shapes that are timed and the shapes that are correctness-tested cannot drift apart. Enforced at
# collection by tests/conftest.py; see fold_cp_ops/testing/kernel_matrix.py for the mechanics.
LAYERNORM = KernelMatrix(
    kernel="layernorm",
    axes=(
        Axis(
            name="N",
            domain=(
                "any positive int; the ONLY constraint is the 16-byte alignment floor (CLAUDE.md). "
                "Not restricted to powers of two, tile multiples, or even numbers -- odd N is "
                "supported at float32 and fails only at 16-bit, where the DSL's 16-bit copy atom "
                "fails IR verification. The TRANSPOSING variant narrows this to a multiple of the "
                "SMEM swizzle atom (16 elements at 16-bit, 32 at fp32) and N <= 1024, which is a "
                "property of that LOAD and is declared as unsupported regions rather than as a "
                "smaller pool -- the same N is supported by the row-major variant"
            ),
            values=(
                # threads_per_row 8 and 16
                16,
                64,
                80,
                96,
                128,
                130,
                160,
                192,
                # threads_per_row 32
                255,
                256,
                257,
                384,
                512,
                528,
                560,
                672,
                704,
                760,
                768,
                784,
                800,
                832,
                864,
                880,
                960,
                999,
                1000,
                1001,
                1002,
                1008,
                1016,
                1024,
                1128,
                2048,
                2688,
                2816,
                3000,
                3072,
                # threads_per_row 64 / 128
                4095,
                4096,
                6136,
                8191,
                8192,
                8200,
                12000,
                12288,
                14336,
                16384,
                # past the num_threads switch: reload_from="smem", then cluster_n > 1
                16640,
                20000,
                32768,
                40000,
                65536,
                131072,
                262144,
            ),
            # tile=128 is the coarsest tile width the ladders produce; big=16384 is where
            # num_threads and reload_from both switch; small=128 is the top of the 16-lane rung.
            facets={
                **int_facets(tile=128, big=16384, small=128),
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
                "any positive int; on the row-major path the row count is a symbolic dim, so it "
                "never recompiles. The TRANSPOSING variant additionally needs M 16-byte aligned "
                "(M is the contiguous mode of the (N, M) TMA view) and recompiles per M, since "
                "the grid and the row guard are both derived from it"
            ),
            # Kept DELIBERATELY short. `single_row` can only ever be satisfied by M=1, so its
            # entropy is 1/len(values) and every value added to this pool pushes it toward the
            # MIN_FACET_ENTROPY floor -- measured: 23 values scores H=0.26 against a floor of 0.25,
            # i.e. one more value would fail an unrelated lock test for a reason nobody would guess.
            # So a transposing cell reuses an existing M wherever one works; the additions are the
            # four non-power-of-two token counts the brief names, the misaligned-M fault value, and
            # the two production counts that were not already here.
            values=(
                1,
                12,
                32,
                37,
                64,
                199,
                384,
                696,
                1000,
                1512,
                4096,
                8192,
                16384,
                65536,
                262144,
                1048576,
                4194304,
                9437184,
            ),
            facets={
                "single_row": lambda v: v == 1,
                "partial_row_tile": lambda v: v % 16 != 0,
                "production_scale": lambda v: v >= 262144,
                # The declared FRONT series, under the name the table uses. Added ALONGSIDE
                # the facet above rather than replacing it: that one is guarding these values
                # today, and a rename would drop a live guard to gain a name. Two names for
                # one set is the cheap side of that trade.
                "workflow_M": lambda v: v in (262144, 1048576, 4194304, 9437184),
            },
        ),
        Axis(
            name="input_dtype",
            domain="float16 / bfloat16 / float32 -- what layernorm_fwd's assert admits",
            values=(torch.bfloat16, torch.float16, torch.float32),
            facets=dtype_facets((torch.bfloat16, torch.float16, torch.float32)),
        ),
        Axis(
            name="eps",
            domain="any positive float; added to the variance before the rsqrt",
            values=(1e-5, 1e-6),
            facets={"1e-5": lambda v: v == 1e-5, "1e-6": lambda v: v == 1e-6},
        ),
        Axis(
            name="has_bias",
            domain="bias is optional; weight is not",
            values=(False, True),
            facets={"with_bias": lambda v: v, "without_bias": lambda v: not v},
        ),
        Axis(
            name="row_mean",
            domain=(
                "any float. NOT a kernel parameter -- it is a property of the INPUT, and it earns "
                "an axis because a padded-tile variance bug is proportional to mean**2 and so is "
                "invisible at the zero mean torch.randn produces"
            ),
            values=(0.0, 100.0, 1000.0, 10000.0),
            facets={
                "zero_mean": lambda v: v == 0.0,
                "non_zero_mean": lambda v: v != 0.0,
                "extreme_mean": lambda v: abs(v) >= 1000.0,
            },
        ),
        Axis(
            name="arg_fault",
            domain=(
                "a malformed TENSOR ARGUMENT, or 'none'. The values name a CALLER MISTAKE rather "
                "than a kernel configuration, and every non-'none' one must be refused at the front "
                "door with an API-level error. Both a dtype and an extent are in the pool because a "
                "caller can get either wrong on any argument, and which arguments the kernel feeds "
                "to a TMA versus reads with a broadcast load is not visible in its signature"
            ),
            values=("none", "weight_dtype", "weight_extent"),
            facets={
                "well_formed": lambda v: v == "none",
                "bad_dtype": lambda v: v.endswith("_dtype"),
                "bad_extent": lambda v: v.endswith("_extent"),
            },
        ),
        Axis(
            name="transpose",
            domain=(
                "which LOAD the kernel uses, and therefore which memory layout x must arrive in. "
                "False -> the cp.async row-major path (LayerNorm); True -> the TMA+swizzle "
                "transposing path (LayerNormTransposeSm90), whose x is (M, N) LayoutLeft, stride "
                "(1, M). Both compute the same normalize + affine, so this is a genuine kernel "
                "config axis and not a second op"
            ),
            values=(False, True),
            facets={
                "row_major_in": lambda v: not v,
                "layout_left_in": lambda v: v,
            },
        ),
        Axis(
            name="do_transpose",
            domain=(
                "transpose=True only -- how the per-thread reduction fragment is built from the "
                "staged tile. 'SMEM' round-trips through an sT tile; 'shuffle' gathers straight "
                "out of sX (bit-identical output, no cross-lane exchange -- the transpose is an "
                "index permutation); 'ldmatrix' is a documented blocker; 'auto' resolves by N. "
                "Ignored with transpose=False, where passing it is refused at the front door"
            ),
            values=("SMEM", "shuffle", "ldmatrix", "auto"),
            facets={
                "explicit": lambda v: v != "auto",
                "resolved_by_size": lambda v: v == "auto",
                "no_smem_round_trip": lambda v: v in ("shuffle", "ldmatrix"),
            },
        ),
        Axis(
            name="transpose_fault",
            domain=(
                "a caller mistake specific to the TRANSPOSING front door, or 'none'. These are "
                "shape and flag faults rather than malformed tensor arguments, which is why they "
                "are a separate axis from 'arg_fault': they name a request the transposing LOAD "
                "cannot serve, not an argument whose dtype or extent is wrong. Every non-'none' "
                "value must be refused with an API-level error before any kernel is built"
            ),
            values=(
                "none",
                "feature_below_swizzle_atom",
                "feature_above_smem_cap",
                "token_misaligned",
                "row_major_input",
                "stats_requested",
                "affine_absent",
            ),
            facets={
                "well_formed": lambda v: v == "none",
                "bad_feature_extent": lambda v: v.startswith("feature_"),
                "bad_token_extent": lambda v: v == "token_misaligned",
                "bad_input_layout": lambda v: v == "row_major_input",
                "unsupported_output": lambda v: v == "stats_requested",
                "absent_operand": lambda v: v == "affine_absent",
            },
        ),
    ),
    # MEASURED, not inferred (M=8, row mean 100, rstd vs the fp32 reference): float32 computes
    # every N from 1 up -- including N=1, 3, 5, i.e. 4, 12 and 20 bytes, all below the 16-byte
    # figure CLAUDE.md names as the repo-wide floor. bfloat16 computes every EVEN N, including
    # N=2 (4 bytes) and N=6 (12 bytes). The real constraint is neither alignment nor parity but
    # the copy-atom width: vecsize = gcd(N, 4), and vecsize * dtype.width must reach 32 bits.
    # Nothing in the swept space returns a wrong answer -- everything either computes or raises.
    #
    # Every region below pins `transpose`, `do_transpose` and `transpose_fault`, and the transposing
    # ones also pin `arg_fault`. That is not decoration: `parametrize_unsupported` sweeps the FULL
    # cross product of the named axes and emits a cell for every combo a region matches, so a region
    # that leaves an axis free multiplies by that axis's whole pool. Pinning a representative cell
    # is the same idiom the `arg_fault` regions already use -- the fault is what is under test, and
    # re-running it at 55 feature extents proves nothing new.
    # Normalizes over a row (so the row mean conditions the variance) and, in the
    # transpose variant, moves data between two layouts.
    computes=("row_reduction", "data_movement"),
    unsupported=(
        Unsupported(
            where=lambda N, input_dtype, transpose, do_transpose, transpose_fault: (
                not transpose
                and do_transpose == "auto"
                and transpose_fault == "none"
                and N % 2 == 1
                and input_dtype in (torch.bfloat16, torch.float16)
            ),
            raises=ValueError,
            match=r"unsupported N=\d+ .*copy would be 16 bits",
            reason=(
                "odd N at a 16-bit dtype gives vecsize = gcd(N, 4) = 1, so the copy would be a "
                "16-bit atom -- which fails IR verification inside cutlass-dsl. The kernel refuses "
                "it at the front door rather than letting that ICE escape: the region exists to "
                "force that guard, not to bless the ICE. float32 takes the same path at 32 bits "
                "and works, which is why this is keyed on the dtype and not on N alone. Pinned to "
                "the row-major load because that is where this atom is chosen; the transposing "
                "load refuses odd N earlier, and for a different reason (the swizzle atom)"
            ),
        ),
        Unsupported(
            where=lambda N, input_dtype, arg_fault, transpose, do_transpose, transpose_fault: (
                not transpose
                and do_transpose == "auto"
                and transpose_fault == "none"
                and N == 1024
                and input_dtype == torch.bfloat16
                and arg_fault == "weight_dtype"
            ),
            raises=ValueError,
            match=r"weight must be torch.float32",
            reason=(
                "the gain is applied in fp32 inside the normalize, so a 16-bit one loses more precision than the normalize it scales. It was an `assert` until measured -- and `python -O` strips asserts, so under -O it was not a check at all"
            ),
        ),
        Unsupported(
            where=lambda N, input_dtype, arg_fault, transpose, do_transpose, transpose_fault: (
                not transpose
                and do_transpose == "auto"
                and transpose_fault == "none"
                and N == 1024
                and input_dtype == torch.bfloat16
                and arg_fault == "weight_extent"
            ),
            raises=ValueError,
            match=r"weight must be",
            reason=(
                "a gain shorter than N is read as if it were the right length, scaling the tail of every row with whatever follows it in memory"
            ),
        ),
        # ── the transposing variant's own gates ────────────────────────────────────────────────
        Unsupported(
            where=lambda N, input_dtype, arg_fault, transpose, do_transpose, transpose_fault: (
                transpose
                and do_transpose == "auto"
                and arg_fault == "none"
                and N == 1024
                and input_dtype == torch.bfloat16
                and transpose_fault == "feature_below_swizzle_atom"
            ),
            raises=ValueError,
            match=r"must be a multiple of \d+ \(the transpose swizzle atom",
            reason=(
                "the transposing load stages x through a 128-B-swizzled SMEM tile whose atom is 16 "
                "elements wide at 16-bit and 32 at fp32. Below that floor the CUTLASS DSL backend "
                "does not raise -- it CORRUPTS ITS HEAP during MLIR lowering, which is a crash with "
                "no stack pointing at the caller. The region exists to force a front-door refusal "
                "that names N and the dtype instead"
            ),
        ),
        Unsupported(
            where=lambda N, input_dtype, arg_fault, transpose, do_transpose, transpose_fault: (
                transpose
                and do_transpose == "auto"
                and arg_fault == "none"
                and N == 1024
                and input_dtype == torch.bfloat16
                and transpose_fault == "feature_above_smem_cap"
            ),
            raises=ValueError,
            match=r"> 1024; the transposing load stages a whole row in SMEM",
            reason=(
                "one CTA stages a WHOLE row of N in shared memory so the reduction sees it "
                "contiguously; past N=1024 there is no tile that fits, at any blk_m. This is the "
                "one place the transposing variant is genuinely narrower than the row-major one, "
                "which is why it is a declared region and not a smaller N pool"
            ),
        ),
        Unsupported(
            where=lambda N, input_dtype, arg_fault, transpose, do_transpose, transpose_fault: (
                transpose
                and do_transpose == "auto"
                and arg_fault == "none"
                and N == 1024
                and input_dtype == torch.bfloat16
                and transpose_fault == "token_misaligned"
            ),
            raises=ValueError,
            match=r"must be 16-byte aligned for the \(N, M\) TMA view",
            reason=(
                "M is the CONTIGUOUS mode of the (N, M) view the TMA reads, so the repo's one "
                "permitted shape constraint -- 16-byte alignment -- lands on M here rather than on "
                "the feature axis. An unaligned M builds a TMA descriptor the hardware rejects at "
                "launch, several frames from the caller"
            ),
        ),
        Unsupported(
            where=lambda N, input_dtype, arg_fault, transpose, do_transpose, transpose_fault: (
                transpose
                and do_transpose == "auto"
                and arg_fault == "none"
                and N == 1024
                and input_dtype == torch.bfloat16
                and transpose_fault == "row_major_input"
            ),
            raises=ValueError,
            match=r"must be LayoutLeft",
            reason=(
                "THE most dangerous mistake on this path: a row-major x has the right shape and "
                "dtype, so nothing downstream objects -- it is simply READ as if it were "
                "LayoutLeft, and the kernel returns a plausible wrong answer. The stride is the "
                "only thing that distinguishes the two, so it has to be checked at the door"
            ),
        ),
        Unsupported(
            where=lambda N, input_dtype, arg_fault, transpose, do_transpose, transpose_fault: (
                transpose
                and do_transpose == "auto"
                and arg_fault == "none"
                and N == 1024
                and input_dtype == torch.bfloat16
                and transpose_fault == "stats_requested"
            ),
            raises=ValueError,
            match=r"return_rstd / return_mean are not available",
            reason=(
                "the transposing kernel writes only the normalized output -- there is no rstd/mean "
                "store in its epilogue. Honouring the flag by returning the empty buffers would be "
                "a silent wrong answer, so the flag combination is refused instead"
            ),
        ),
        Unsupported(
            where=lambda N, input_dtype, arg_fault, transpose, do_transpose, transpose_fault: (
                transpose
                and do_transpose == "auto"
                and arg_fault == "none"
                and N == 1024
                and input_dtype == torch.bfloat16
                and transpose_fault == "affine_absent"
            ),
            raises=ValueError,
            match=r"weight and bias are both REQUIRED",
            reason=(
                "the transposing kernel's epilogue reads mW and mB unconditionally -- there is no "
                "const_expr branch for an absent affine, unlike the row-major variant where bias is "
                "genuinely optional. An absent one would be read from unallocated memory, so it is "
                "refused rather than silently zero-filled. Making it optional is a KERNEL change "
                "and would break byte-identity with the source this was ported from"
            ),
        ),
        Unsupported(
            where=lambda N, input_dtype, arg_fault, transpose, do_transpose, transpose_fault: (
                transpose
                and do_transpose == "ldmatrix"
                and arg_fault == "none"
                and transpose_fault == "none"
                and N == 1024
                and input_dtype == torch.bfloat16
            ),
            raises=NotImplementedError,
            match=r"do_transpose='ldmatrix' is not implemented",
            reason=(
                "LDSM.x4.trans delivers a fragment in which ONE lane's 8 registers span 4 distinct "
                "tokens, which cannot feed row_reduce's 'one lane group owns one token' layout "
                "without a from-scratch two-stage warp reduction. The mode is declared and refused "
                "rather than quietly absent, because 'shuffle' already removes the same SMEM "
                "round-trip it was for and is bit-identical to 'SMEM'"
            ),
        ),
    ),
)


@LAYERNORM.parametrize(
    "N",
    "M",
    "input_dtype",
    "eps",
    only={
        "N": [
            64,
            128,
            256,
            384,
            512,
            760,
            1002,
            1024,
            1128,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
        ],
        "M": [1, 37, 199],
    },
    because=(
        "the broad correctness grid, so N is the ladder-spanning subset: it crosses every "
        "threads_per_row rung, every cluster_n rung, both sides of the num_threads switch, and "
        "both vecsize values. The specialist extents -- odd N, the delay_w_load boundary cells, "
        "the tiny rungs -- have dedicated tests below, and carrying them here would multiply "
        "through 3 dtypes x 3 M x 2 eps for no new signal. M is a symbolic dim so it never "
        "recompiles; 1 and 37 cover single-row and partial-row-tile."
    ),
)
def test_layernorm_forward(M, N, input_dtype, eps):
    """Test LayerNorm forward pass against reference implementation.

    **The N list is not a ladder of powers of two.** It spans every ``_threads_per_row`` rung
    (8/16/32/64/128/256), every ``cluster_n`` rung (1 through 16, at 16384/32768/65536/131072/
    262144), both sides of the ``_num_threads`` switch at 16384, and every ``vecsize`` the copy can
    resolve to (4 almost everywhere, 2 at N=1002). **128, 256, 384 and 512 are the feature dims the
    TriMul front actually calls this kernel with** -- they earn their place by being production
    shapes, not by being round.

    **Zero-mean inputs only** -- ``torch.randn``. That is a real blind spot, not an oversight to
    leave implicit: a padded-tile variance bug is invisible at zero mean because the spurious term
    is proportional to ``mean**2``. It lived here undetected across every cell of this grid,
    including the off-grid N=760/1128 ones. ``test_layernorm_padded_tile_variance`` below is the
    test that actually covers that axis; keep it in step with this grid's N values.
    """
    device = "cuda"

    # tolerance depends on precision
    if input_dtype == torch.bfloat16:
        atol = 1e-2
        rtol = 1e-2
    elif input_dtype == torch.float16:
        atol = 1e-3
        rtol = 1e-3
    else:
        atol = 1e-4
        rtol = 1e-4

    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)

    # pure‐PyTorch refs
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()

    out, rstd, mean = layernorm_fwd(x, weight, eps=eps, return_rstd=True, return_mean=True)
    out_ref = layernorm_ref(x_ref, weight_ref, eps=eps)
    rstd_ref_val = layernorm_rstd_ref(x_ref, eps=eps)
    mean_ref_val = layernorm_mean_ref(x_ref)

    # shapes & dtypes
    assert out.shape == x.shape
    assert out.dtype == input_dtype
    assert rstd.shape == (M,) and rstd.dtype == torch.float32
    assert mean.shape == (M,) and mean.dtype == torch.float32

    # numeric check
    assert_elementwise(out, out_ref, tolerance_bound(out_ref, atol, rtol), what="out")
    assert_elementwise(rstd, rstd_ref_val, tolerance_bound(rstd_ref_val, 6e-4, 6e-4), what="rstd")
    assert_elementwise(mean, mean_ref_val, tolerance_bound(mean_ref_val, 6e-4, 6e-4), what="mean")


@LAYERNORM.parametrize(
    "N",
    "M",
    "input_dtype",
    "has_bias",
    "eps",
    only={"N": [256, 512, 1024, 2048], "M": [1, 199, 8192]},
    because=(
        "bias PRESENCE is the axis under test, so the shape axes stay small and tile-aligned to "
        "keep the cell count down; M=8192 is here for a large grid. The shape ladder itself is "
        "covered by test_layernorm_forward."
    ),
)
def test_layernorm_forward_with_bias(M, N, input_dtype, has_bias, eps):
    """Forward correctness with fp32 weight and optional fp32 bias.

    Bias PRESENCE is the axis under test, so the shape axes stay small and tile-aligned; output,
    rstd and mean are all compared, at the same per-dtype tolerances the rest of this file uses.
    """
    device = "cuda"

    if input_dtype == torch.bfloat16:
        atol = rtol = 1e-2
    elif input_dtype == torch.float16:
        atol = rtol = 1e-3
    else:
        atol = rtol = 1e-4

    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype)
    weight = torch.randn(N, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32) if has_bias else None

    out, rstd, mean = layernorm_fwd(
        x, weight, bias=bias, eps=eps, return_rstd=True, return_mean=True
    )

    # Reference. layernorm_ref doesn't take bias, so apply it via F.layer_norm
    # directly when bias is present; otherwise reuse the shared ref.
    x_f32 = x.float()
    if bias is None:
        out_ref = layernorm_ref(x, weight, eps=eps)
    else:
        out_ref = torch.nn.functional.layer_norm(x_f32, weight.shape, weight, bias, eps).to(x.dtype)
    rstd_ref_val = layernorm_rstd_ref(x, eps=eps)
    mean_ref_val = layernorm_mean_ref(x)

    assert out.shape == x.shape
    assert out.dtype == input_dtype
    assert rstd.shape == (M,) and rstd.dtype == torch.float32
    assert mean.shape == (M,) and mean.dtype == torch.float32

    assert_elementwise(out, out_ref, tolerance_bound(out_ref, atol, rtol), what="out")
    assert_elementwise(rstd, rstd_ref_val, tolerance_bound(rstd_ref_val, 6e-4, 6e-4), what="rstd")
    assert_elementwise(mean, mean_ref_val, tolerance_bound(mean_ref_val, 6e-4, 6e-4), what="mean")


@matrix_exempt(
    "asserts a jit_cache KEY property -- that dtype and N recompile while M does not -- so a "
    "shape sweep would only re-derive the same key repeatedly"
)
def test_layernorm_compile_cache():
    """The LayerNorm jit cache keys on dtype/N and reuses across batch sizes."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6

    _compile_layernorm_fwd.cache_clear()
    assert _compile_layernorm_fwd.cache_info().currsize == 0

    x1 = torch.randn(M, N, device=device, dtype=torch.float16)
    weight1 = torch.randn(N, device=device, dtype=torch.float32)
    layernorm_fwd(x1, weight1, eps=eps)
    assert _compile_layernorm_fwd.cache_info().currsize == 1

    # Same shape/dtype reuses the cache entry.
    layernorm_fwd(
        torch.randn(M, N, device=device, dtype=torch.float16),
        torch.randn(N, device=device, dtype=torch.float32),
        eps=eps,
    )
    assert _compile_layernorm_fwd.cache_info().currsize == 1

    # Different batch size reuses the cache entry (batch is a dynamic dim).
    layernorm_fwd(
        torch.randn(M * 2, N, device=device, dtype=torch.float16),
        torch.randn(N, device=device, dtype=torch.float32),
        eps=eps,
    )
    assert _compile_layernorm_fwd.cache_info().currsize == 1

    # Different N creates a new cache entry.
    layernorm_fwd(
        torch.randn(M, N * 2, device=device, dtype=torch.float16),
        torch.randn(N * 2, device=device, dtype=torch.float32),
        eps=eps,
    )
    assert _compile_layernorm_fwd.cache_info().currsize == 2

    # Different dtype creates a new cache entry.
    layernorm_fwd(
        torch.randn(M, N, device=device, dtype=torch.float32),
        torch.randn(N, device=device, dtype=torch.float32),
        eps=eps,
    )
    assert _compile_layernorm_fwd.cache_info().currsize == 3


@pytest.mark.parametrize("return_rstd", [True, False])
@pytest.mark.parametrize("return_mean", [True, False])
@matrix_exempt(
    "exercises the return-flag API surface at one fixed shape; the flags select which outputs are "
    "allocated, not any kernel config axis"
)
def test_layernormnorm_return_rstd_option(return_rstd, return_mean):
    """Test that return_rstd option works correctly."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6

    x = torch.randn(M, N, device=device, dtype=torch.float16)
    weight = torch.randn(N, device=device, dtype=torch.float32)

    if return_rstd and return_mean:
        out, rstd, mean = layernorm_fwd(x, weight, eps=eps, return_rstd=True, return_mean=True)
        assert out.shape == (M, N)
        assert rstd.shape == (M,)
        assert rstd.dtype == torch.float32
        assert mean.shape == (M,)
        assert mean.dtype == torch.float32
    elif return_rstd and not return_mean:
        out, rstd = layernorm_fwd(x, weight, eps=eps, return_rstd=True, return_mean=False)
        assert out.shape == (M, N)
        assert rstd.shape == (M,)
        assert rstd.dtype == torch.float32
    elif not return_rstd and return_mean:
        out, mean = layernorm_fwd(x, weight, eps=eps, return_rstd=False, return_mean=True)
        assert out.shape == (M, N)
        assert mean.shape == (M,)
        assert mean.dtype == torch.float32
    else:
        out = layernorm_fwd(x, weight, eps=eps, return_rstd=False, return_mean=False)
        assert out.shape == (M, N)
        assert isinstance(out, torch.Tensor)


@matrix_exempt(
    "asserts the argument-validation raises, every one of which fires before a shape ever reaches "
    "the kernel"
)
def test_layernorm_input_validation():
    """Test input validation and error handling."""
    device = "cuda"

    # Test 3D input (should fail)
    x_3d = torch.randn(2, 32, 1024, device=device, dtype=torch.float16)
    weight = torch.randn(1024, device=device, dtype=torch.float32)

    # ValueError, not AssertionError: `python -O` strips asserts, so an assert-based guard is not a
    # guard under -O. Every front-door refusal in this package is an API-level error for that reason.
    with pytest.raises(ValueError, match=r"x must be 2-D"):
        layernorm_fwd(x_3d, weight)

    # Test weight dimension mismatch -- now caught AT THE DOOR, naming the argument and the axis,
    # rather than several frames later as a symbolic-shape mismatch naming `mW.shape[0]`.
    x = torch.randn(32, 1024, device=device, dtype=torch.float16)
    weight_wrong = torch.randn(512, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"weight must be"):
        layernorm_fwd(x, weight_wrong)

    # Test CPU tensors (should fail)
    x_cpu = torch.randn(32, 1024, dtype=torch.float16)
    weight_cpu = torch.randn(1024, dtype=torch.float32)

    # with pytest.raises(AssertionError, match="Tensors must be on CUDA device"):
    # With torch.library custom op, this now fails with NotImplementedError
    with pytest.raises(NotImplementedError):
        layernorm_fwd(x_cpu, weight_cpu)

    # Test unsupported dtype
    x = torch.randn(32, 1024, device=device, dtype=torch.float64)
    weight = torch.randn(1024, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"unsupported dtype for x"):
        layernorm_fwd(x, weight)

    # Test wrong weight dtype
    x = torch.randn(32, 1024, device=device, dtype=torch.float16)
    weight_wrong_dtype = torch.randn(1024, device=device, dtype=torch.float16)

    with pytest.raises(ValueError, match=r"weight must be torch.float32"):
        layernorm_fwd(x, weight_wrong_dtype)


# ── the padded-tile variance: a non-zero row mean must not leak through the masked tail ────────
@LAYERNORM.parametrize(
    "N",
    "M",
    "input_dtype",
    "row_mean",
    only={
        "N": [16, 130, 384, 760, 1000, 1002, 1016, 1128, 3000, 12000, 20000, 40000],
        "M": [1, 37, 1000],
        "input_dtype": [torch.float32, torch.bfloat16],
        "row_mean": [0.0, 100.0],
    },
    because=(
        "float16 is dropped because it shares bfloat16's 16-bit path exactly and would double the "
        "cell count for no new geometry. The extreme row means (1e3, 1e4) belong to the numerics "
        "probe below, which needs an fp64 oracle to be meaningful. ODD N is excluded because 2 of "
        "the 3 dtypes cannot compile it; it is covered at float32 by "
        "test_layernorm_odd_feature_extent_fp32, which also carries a non-zero row mean."
    ),
)
def test_layernorm_padded_tile_variance(N, M, input_dtype, row_mean):
    """Variance is correct when the CTA tile is padded AND the row mean is non-zero.

    **The regression test for a silent wrong-output bug** (docs/refactor_and_fix.md §3.3). The
    predicated ``cp.async`` zero-fills the masked tail of the staged tile. Zero is the identity for
    the mean's ADD reduction, so pass 1 was always right; pass 2 reduces ``(x - mean)**2`` over the
    whole tile, so every masked lane contributed ``mean**2`` instead of 0 and inflated the variance
    by ``(tile - N) * mean**2 / N``. Measured before the fix at ``row_mean=100``: rstd off by
    **9.4e-01 relative** -- not a tolerance failure, a wrong answer.

    Both axes are required to trigger it, which is exactly why it survived 598 tests: the N=384
    row has no padding, and the ``row_mean=0.0`` rows are what every other test in this file uses.

    **Each axis is chosen to separate a distinct mechanism, not to pile on cells.**

    * *N* walks the pad fraction from 50% (N=16, tile 32) down to a few columns, and crosses every
      downstream path the masked value has to survive: ``vecsize`` 4 vs 2, one row per tile
      (N=12000) vs sixteen (N=16), a re-read of x from SMEM between the passes (N=20000), and a
      reduction that crosses CTAs through distributed SMEM (N=40000). The fix has to hold *after*
      the reload, not just at the first load.
    * *M* is not decoration: 1 and 37 do not fill the tile's row extent (``tiler_m`` is 1/4/8/16
      across this N list), so the rows above ``M`` hold **uninitialized SMEM** and take part in the
      same reductions. Their results are discarded by the ``row < shape[0]`` guard, and this is
      what pins that -- a masked-tail fill that touched the row guard would corrupt real rows here.
    * *row_mean=0.0* is the control that the fix did not break the case everything else tests.

    Tolerances are tight on purpose. rstd is compared at 1e-4 relative, which the fixed kernel
    clears by three orders of magnitude (worst measured 3.4e-07), so the gap between pass and the
    pre-fix 9.4e-01 is unmistakable.
    """
    device, eps = "cuda", 1e-6
    atol = rtol = 1e-2 if input_dtype == torch.bfloat16 else 1e-4

    torch.manual_seed(0)
    x = (torch.randn(M, N, device=device, dtype=torch.float32) + row_mean).to(input_dtype)
    weight = torch.randn(N, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)

    out, rstd, mean = layernorm_fwd(
        x, weight, bias=bias, eps=eps, return_rstd=True, return_mean=True
    )
    out_ref = torch.nn.functional.layer_norm(x.float(), (N,), weight, bias, eps).to(input_dtype)

    rstd_ref_val, mean_ref_val = layernorm_rstd_ref(x, eps=eps), layernorm_mean_ref(x)
    assert_elementwise(rstd, rstd_ref_val, tolerance_bound(rstd_ref_val, 1e-4, 1e-4), what="rstd")
    assert_elementwise(mean, mean_ref_val, tolerance_bound(mean_ref_val, 1e-4, 1e-4), what="mean")
    assert_elementwise(out, out_ref, tolerance_bound(out_ref, atol, rtol), what="out")


@LAYERNORM.parametrize("row_mean")
def test_layernorm_variance_error_does_not_scale_with_row_mean(row_mean):
    """The masked tail contributes EXACTLY zero, so accuracy is flat in the row mean.

    **This is the test that pins the *choice* of fix, not merely that a fix exists.** The defect's
    signature was an error growing as ``mean**2``; two of the three alternatives that were
    considered leave a residue with that same shape and would pass the test above (which only ever
    reaches mean=100) while failing here:

    * filling the tile's tail with ``mean`` cast to the element dtype leaves
      ``(dtype(mean) - mean)**2`` per masked lane -- an error proportional to ``mean**2``;
    * subtracting the known ``n_pad * mean**2`` after the reduction leaves an fp32 rounding residue
      of ``~1e-7 * (n_pad/N) * (mean/std)**2`` -- again proportional to ``mean**2``.

    Masking the *centred* values is exact instead of merely small, because an fp32 ``mean - mean``
    is 0 with no residue at all. So the assertion is that the error curve is FLAT: the same 1e-4
    bound holds four decades of row mean apart. The top of the range is deliberate -- at mean=1e4
    the two rejected forms are already off by ~1e-2 and ~1e-3 respectively.

    float32 only, and N=1000 is fixed: this is a numerics probe, not a shape probe, and a 16-bit
    input at mean=1e4 would quantize x itself far above the signal being measured.
    """
    device, M, N, eps = "cuda", 64, 1000, 1e-6
    torch.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=torch.float32) + row_mean
    weight = torch.ones(N, device=device, dtype=torch.float32)

    _, rstd = layernorm_fwd(x, weight, eps=eps, return_rstd=True)
    # fp64 oracle -- independent of the kernel's fp32 accumulation order AND of the reference's.
    xd = x.double()
    ref = 1.0 / torch.sqrt(((xd - xd.mean(-1, keepdim=True)) ** 2).mean(-1) + eps)
    assert_elementwise(
        rstd.double(),
        ref,
        1e-4 * ref.abs(),
        what=(
            f"row_mean={row_mean:g} rstd. An error that grows with the row mean is the padded-tail "
            f"signature -- the masked lanes are contributing something, not zero"
        ),
    )


@LAYERNORM.parametrize(
    "N",
    only={"N": [255, 257, 999, 1001, 4095, 8191]},
    because=(
        "the odd extents, which are float32-only and therefore their own test: the 16-bit copy "
        "atom an odd N requires fails IR verification in the DSL, pinned by the raises-test below"
    ),
)
def test_layernorm_odd_feature_extent_fp32(N):
    """An ODD feature extent is supported at float32, including with a non-zero row mean.

    Nothing in the algorithm needs an even N: ``vecsize = gcd(N, 128 // width)`` degrades to 1 and
    the copy falls back to scalar 32-bit accesses. These are the most adversarial shapes the tile
    machinery sees -- every one is off-grid, none is a power of two, and each lands on a different
    ``_threads_per_row`` rung (255/257 -> 32, 999/1001 -> 32, 4095 -> 64, 8191 -> 128).

    257 and 8191 are prime, so no vectorization is available at any width; 8191 is also the
    tightest padding this kernel ever sees -- its tile is 8192, so **exactly one** column is
    masked, which is the case an off-by-one in the predicate would survive everywhere else.

    They are float32-only because the 16-bit copy atom the same path would need at bf16/fp16 fails
    IR verification in the DSL; that boundary is pinned separately below.
    """
    device, M, eps = "cuda", 64, 1e-6
    torch.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=torch.float32) + 100.0
    weight = torch.randn(N, device=device, dtype=torch.float32)

    out, rstd = layernorm_fwd(x, weight, eps=eps, return_rstd=True)
    out_ref = torch.nn.functional.layer_norm(x, (N,), weight, None, eps)

    rstd_ref_val = layernorm_rstd_ref(x, eps=eps)
    assert_elementwise(rstd, rstd_ref_val, tolerance_bound(rstd_ref_val, 1e-4, 1e-4), what="rstd")
    assert_elementwise(out, out_ref, tolerance_bound(out_ref, 1e-4, 1e-4), what="out")


@LAYERNORM.parametrize(
    "N",
    "input_dtype",
    "row_mean",
    only={
        "N": [384, 1000, 20000],
        "input_dtype": [torch.float32, torch.bfloat16],
        "row_mean": [0.0, 100.0],
    },
    because=(
        "reaches the registered op directly, since layernorm_fwd never passes a residual, so it "
        "keeps one aligned N, one padded, and one that re-reads x from SMEM. float16 shares "
        "bfloat16's path; the extreme means belong to the numerics probe."
    ),
)
def test_layernorm_residual_path(N, input_dtype, row_mean):
    """The fused residual add: ``out = LN(x + residual)``, ``residual_out = x + residual``.

    ``layernorm_fwd`` never passes a residual, so this reaches the op directly -- but the op is
    **registered and invocable** (``fold_cp_ops::_layernorm_fwd``), which under CLAUDE.md makes it
    a path that must work rather than one that merely exists. It had no coverage at all, and the
    masked-tail fix runs downstream of it: the reduction consumes ``x + residual``, and on a padded
    tile BOTH operands are zero-filled in the tail, so the fill has to neutralize their sum.

    The sum is formed in **float32** before normalizing, and ``residual_out`` is that fp32 sum cast
    down -- not the other way round. The distinction is worth asserting: rounding first and then
    normalizing would be a different (and, at bf16, visibly worse) answer, and nothing in the type
    signature would catch the swap.

    N=384 is the aligned control, 1000 is padded, 20000 is padded *and* re-reads x from SMEM
    between the two passes -- which is the case where the residual has to be re-added too.
    """
    device, M, eps = "cuda", 37, 1e-6
    atol = rtol = 1e-2 if input_dtype == torch.bfloat16 else 1e-4

    torch.manual_seed(0)
    x = (torch.randn(M, N, device=device, dtype=torch.float32) + row_mean).to(input_dtype)
    res = (torch.randn(M, N, device=device, dtype=torch.float32) + row_mean).to(input_dtype)
    weight = torch.randn(N, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)
    out, res_out = torch.empty_like(x), torch.empty_like(x)
    rstd = torch.empty(M, device=device, dtype=torch.float32)
    mean = torch.empty(M, device=device, dtype=torch.float32)

    _ln._layernorm_fwd(x, weight, out, bias, rstd, mean, res, res_out, eps)

    x_eff = x.float() + res.float()  # the kernel accumulates in fp32, then casts on the way out
    res_ref = x_eff.to(input_dtype)
    rstd_ref_val, mean_ref_val = layernorm_rstd_ref(x_eff, eps=eps), layernorm_mean_ref(x_eff)
    out_ref = torch.nn.functional.layer_norm(x_eff, (N,), weight, bias, eps).to(input_dtype)
    assert_elementwise(res_out, res_ref, tolerance_bound(res_ref, atol, rtol), what="res_out")
    assert_elementwise(rstd, rstd_ref_val, tolerance_bound(rstd_ref_val, 1e-4, 1e-4), what="rstd")
    assert_elementwise(mean, mean_ref_val, tolerance_bound(mean_ref_val, 1e-4, 1e-4), what="mean")
    assert_elementwise(out, out_ref, tolerance_bound(out_ref, atol, rtol), what="out")


#: The transposing gates, as (fault name -> how the test body BREAKS an otherwise-valid call).
#: Kept beside the test rather than in the matrix because a fault is not a configuration -- the
#: matrix declares WHICH combo must raise, this says how to construct it. Each entry returns the
#: kwargs overriding the well-formed ``(M, N, return_rstd, row_major)`` baseline.
_TRANSPOSE_FAULT_RECIPE = {
    # 1018 % 16 == 10, so it is below the bf16 swizzle atom AND still under the N<=1024 cap --
    # which matters, because the cap is checked after the atom and would otherwise fire first.
    "feature_below_swizzle_atom": {"N": 1018},
    "feature_above_smem_cap": {"N": 2048},
    "token_misaligned": {"M": 12},  # 12 % 8 == 4 at bfloat16
    "row_major_input": {"row_major": True},
    "stats_requested": {"return_rstd": True},
    "affine_absent": {"bias": None},
}


@LAYERNORM.parametrize_unsupported(
    "N", "input_dtype", "arg_fault", "transpose", "do_transpose", "transpose_fault"
)
def test_layernorm_unsupported_combos_raise(
    N,
    input_dtype,
    arg_fault,
    transpose,
    do_transpose,
    transpose_fault,
    expected_error,
    expected_match,
):
    """Every combo the matrix declares UNSUPPORTED raises, rather than returning a wrong answer.

    **The raise must come from the kernel's own front door**, not from wherever the failure lands.
    ``expected_error`` is an API-level exception type, so this cannot be satisfied by the cutlass-dsl
    ICE that odd N used to produce -- the region is what forced ``_layernorm_fwd`` to grow a check
    that names N and the dtype. Delete that check and this test fails, which is the point.

    **A raise, not an xfail.** An xfail'd correctness assertion cannot tell "the kernel refused"
    from "the kernel answered wrongly" -- both make the assertion fail, so both satisfy the xfail
    and the suite stays green. Measured on a stand-in: an xfail passes for a lying kernel;
    ``pytest.raises`` fails it with "DID NOT RAISE".

    The cells come from :data:`LAYERNORM`'s ``unsupported`` regions, so declaring a region and
    testing it are the same act -- ``audit_test_module`` fails the module if a region has no test.

    **If a cell here starts failing because the call SUCCEEDED, that is good news**: the toolchain
    grew support. Delete the region from the matrix and let the shapes flow into the normal grid.

    **Two fault families, one test**, because ``parametrize_unsupported`` requires a single sweep
    wide enough to evaluate every declared region. ``arg_fault`` malforms a tensor ARGUMENT on the
    row-major path; ``transpose_fault`` breaks a SHAPE or a FLAG on the transposing path. They are
    mutually exclusive by construction -- every region pins the other axis to its well-formed value.
    """
    torch.manual_seed(0)
    if transpose:
        # "none" here is the ldmatrix region: the request is well-formed, the MODE is refused.
        rec = _TRANSPOSE_FAULT_RECIPE.get(transpose_fault, {})
        M, N = rec.get("M", 64), rec.get("N", N)
        base = torch.randn(M, N, device="cuda", dtype=input_dtype)
        x = base if rec.get("row_major") else _layout_left(base)
        weight = torch.randn(N, device="cuda", dtype=torch.float32)
        with front_door_raises(expected_error, expected_match):
            layernorm_fwd(
                x,
                weight,
                weight if "bias" not in rec else rec["bias"],
                eps=1e-6,
                transpose=True,
                do_transpose=do_transpose,
                return_rstd=rec.get("return_rstd", False),
            )
        return
    x = torch.randn(64, N, device="cuda", dtype=input_dtype)
    weight = torch.randn(N, device="cuda", dtype=torch.float32)
    # One malformed ARGUMENT, built here rather than in the matrix: a fault is not a configuration,
    # so it does not belong to the cell the other axes describe.
    # Applied ONLY on the cell the fault regions claim. Regions match first-wins, so on an odd-N
    # cell the odd-N region owns the expectation -- malforming the weight there would make the
    # kernel raise about the weight while the test asserted the N message.
    faultable = N == 1024 and input_dtype is torch.bfloat16
    if faultable and arg_fault == "weight_dtype":
        weight = weight.to(torch.bfloat16)
    elif faultable and arg_fault == "weight_extent":
        weight = torch.randn(N + 1, device="cuda", dtype=torch.float32)
    with front_door_raises(expected_error, expected_match):
        layernorm_fwd(x, weight, eps=1e-6)


# ── delay_w_load: a PERF knob that must never move a bit of output ─────────────────────────────
def _run_with_forced_delay(x, w, b, delay: bool):
    """Run layernorm_fwd with ``delay_w_load`` pinned, bypassing the register-pressure gate.

    Args:
        x: 2-D CUDA input; row-major, dtype accepted by ``layernorm_fwd``.
        w: fp32 affine weight, shape ``(x.shape[1],)``.
        b: fp32 affine bias, same shape as ``w``.
        delay: The value to force. True fetches weight/bias after the two reductions, False before.

    Returns:
        The output tensor.

    **Disables the persistent disk cache for the duration, and that is load-bearing.**
    ``_compile_layernorm_fwd``'s key is ``(dtypes..., N, has_rstd, has_mean)`` and does NOT include
    ``delay_w_load`` -- correctly so, because in production the gate derives it *from* those same
    inputs, making it a pure function of the key. Forcing it here breaks that invariant: a
    ``delay=False`` artifact compiled at N=12288/16384 would be written to the shared on-disk cache
    under the key production uses, and a later test in the same session (or a later session
    entirely) would silently load the un-gated, 15x-slower kernel.

    That is not hypothetical -- it is how this was found. The perf gate failed on exactly
    N=12288/16384 bf16 and N=16384 fp32 with pre-fix timings (1.18 / 1.78 / 1.70 ms) in a full-suite
    run while passing standalone. Clearing the in-memory ``lru_cache`` alone is NOT enough; the
    ``.o`` on disk outlives the process, and editing a test does not bust the fingerprint (it hashes
    ``fold_cp_ops/**``, not ``tests/**``).
    """
    _compile_layernorm_fwd.cache_clear()
    orig_cls, orig_cache = _ln.LayerNorm, _cache_utils.CACHE_ENABLED
    _cache_utils.CACHE_ENABLED = False

    class Forced(orig_cls):
        def _resolve_delay_w_load(self, elems_per_thread, threads_per_row, even_n, has_w, has_b):
            return delay

    _ln.LayerNorm = Forced
    try:
        return layernorm_fwd(x, w, b, eps=1e-6)
    finally:
        _ln.LayerNorm, _cache_utils.CACHE_ENABLED = orig_cls, orig_cache
        _compile_layernorm_fwd.cache_clear()


@LAYERNORM.parametrize(
    "N",
    "input_dtype",
    only={"N": [1024, 3072, 12288, 16384]},
    because=(
        "the four extents where the gate's decision differs across dtypes: 1024 never fires, 3072 "
        "fires only at float32, 12288/16384 fire at 16-bit. Bit-identity is a property of the "
        "schedule, so more shapes cost time without adding a case."
    ),
)
def test_delay_w_load_is_bit_identical(N, input_dtype):
    """Both weight/bias load positions produce BITWISE identical output.

    ``delay_w_load`` only moves *when* the affine fragments are fetched; ``y = x_hat * w + b`` is
    untouched, so this is a pure scheduling choice with no numerical content. Asserting bitwise
    equality rather than a tolerance is the point: a tolerance would hide a real reordering of the
    arithmetic, which is exactly the hazard a perf knob must not introduce. N=12288 and N=16384 are
    the shapes where the gate fires at bf16; N=3072 is where it fires at float32 and only there
    (``threads_per_row`` = 32), so it also pins that the rung-dependent threshold changed only the
    schedule.
    """
    torch.manual_seed(0)
    M = 256
    x = torch.randn(M, N, device="cuda", dtype=input_dtype)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    early = _run_with_forced_delay(x, w, b, delay=False)
    late = _run_with_forced_delay(x, w, b, delay=True)
    assert_bitwise(
        early,
        late,
        what=f"N={N} {input_dtype}: delay_w_load changed the output — it is a scheduling knob and must not. max|diff| = {(early.float() - late.float()).abs().max().item():.3e}",
    )


@pytest.mark.parametrize(
    "width,threads_per_row,is_even_N,elems_per_thread,expected,cell",
    [
        # bf16/fp16 (width 16): est = 3 * elems; threshold 255 at EVERY rung and BOTH alignments.
        (16, 32, True, 84, False, "N=2688 aligned, est 252, healthy"),
        (16, 32, True, 88, True, "N=2816 aligned, est 264 -> 11.1x"),
        (16, 32, False, 84, False, "N=2624 padded, est 252, healthy"),
        (16, 32, False, 88, True, "N=2752 padded, est 264 -> 12.2x (padding does NOT save bf16)"),
        (16, 64, True, 84, False, "N=5376, est 252, healthy"),
        (16, 64, True, 92, True, "N=5888, est 276 -> 13.0x"),
        (16, 128, True, 80, False, "N=10240, est 240; delaying costs 4%"),
        (16, 128, True, 128, True, "N=16384, est 384 -> 15.9x, the worst cell"),
        # fp32 (width 32): est = 4 * elems. Fires ONLY on a narrow AND aligned tile.
        (32, 32, True, 84, False, "N=2688 aligned, est 336, healthy"),
        (32, 32, True, 88, True, "N=2816 aligned, est 352 -> 6.4x"),
        (32, 32, True, 96, True, "N=3072 aligned, est 384 -> 6.0x"),
        (32, 32, False, 88, False, "N=2752 PADDED, est 352 -- firing here costs 11%"),
        (32, 32, False, 96, False, "N=3000 PADDED, est 384 -- firing here costs 6%"),
        (32, 64, True, 96, False, "N=6144, est 384, other rung -- healthy"),
        (32, 128, True, 112, False, "N=14336, est 448 at the bound; delaying costs 15%"),
        (32, 128, True, 128, True, "N=16384, est 512 -> 8.6x"),
        (32, 128, True, 68, False, "N=8200, est 272; the bf16 bound here would cost 13%"),
    ],
)
@matrix_exempt(
    "tests the pure gate function against measured cells; its axes are a register estimate and "
    "rung geometry, which are DERIVED from shapes rather than being shape axes themselves"
)
def test_delay_w_load_gate_decisions(
    width, threads_per_row, is_even_N, elems_per_thread, expected, cell
):
    """The gate fires exactly where the sweep showed delaying helps, and nowhere else.

    Every row is a measured cell at M=4096 on H100 (docs/refactor_and_fix.md §3.2), chosen in
    **pairs that bracket a boundary** -- 84 vs 88 elements, 112 vs 128, aligned vs padded -- so a
    threshold that drifts in either direction fails here instead of surfacing as a 6x surprise in a
    benchmark much later.

    Two pairs carry the whole design and neither is redundant:

    * **rows 10 and 12** (fp32, tpr=32, est 352, aligned vs padded). Identical register demand,
      opposite verdicts: the aligned tile collapses to 32 registers and gains 6.4x from delaying,
      the padded one is healthy and *loses* 11%. The padding is what keeps ptxas from giving up.
    * **rows 3 and 4** (bf16, tpr=32, est 264, aligned vs padded). Both fire -- at 16-bit the
      padding does NOT rescue the allocator. So the alignment axis is a float32-only distinction
      and must not be generalized to the 16-bit entries.
    """
    dtype = {16: cutlass.BFloat16, 32: cutlass.Float32}[width]
    k = _ln.LayerNorm(dtype, 1024)
    got = k._resolve_delay_w_load(elems_per_thread, threads_per_row, is_even_N, True, True)
    assert got is expected, f"{cell}: gate returned {got}, expected {expected}"


@matrix_exempt(
    "tests the pure gate function's handling of absent affine operands; no kernel shape axes apply"
)
def test_delay_w_load_ignores_absent_affine_fragments():
    """A weight/bias that does not exist cannot be spilled, so it must not raise the estimate.

    Without this, a bias-free call at a large feature width would delay a load it does not have,
    paying the latency-hiding cost for nothing.
    """
    k = _ln.LayerNorm(cutlass.BFloat16, 1024)
    # est with both = 3*100 = 300 > 255 -> fires; with neither = 1*100 = 100 -> must not.
    assert k._resolve_delay_w_load(100, 32, True, True, True) is True
    assert k._resolve_delay_w_load(100, 32, True, False, False) is False


@matrix_exempt(
    "asserts the threshold table is total over the accepted dtype widths; a shape sweep cannot "
    "observe a missing key"
)
def test_delay_w_threshold_table_covers_supported_widths():
    """Every (dtype, rung) the gate can look up has a threshold; a gap would be a KeyError.

    The lookup key is ``(width, threads_per_row <= 32)``, so both halves of the rung axis have to
    be present for every accepted dtype -- a partial table would raise from inside ``@cute.jit``
    tracing, i.e. at the first call on an unlucky shape rather than at import.
    """
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        width = torch.empty(0, dtype=dt).element_size() * 8
        for narrow in (True, False):
            assert (width, narrow) in _DELAY_W_EST_REG_THRESHOLD, (
                f"{dt} (width {width}, threads_per_row<=32 is {narrow}) has no threshold"
            )


@LAYERNORM.parametrize(
    "M",
    only={"M": (262144,)},
    because=(
        "the subject is the PRODUCTION token counts specifically: the rest of the pool is covered "
        "by the grids above. ONE value, not the four: 262144 is what the production_scale facet "
        "is defined at, and the larger ones allocate 2-19 GB to re-prove the same point on a GPU "
        "the rest of the suite is also using. The transposing variant sweeps all four in "
        "test_transpose_production_token_counts, which it must -- M is in ITS compile key"
    ),
)
def test_production_scale_token_counts_are_correct(M):
    """The largest declared token counts run, at a narrow feature width to keep the allocation sane.

    Declared in the pool because production runs there; without a test that reaches it, the facet
    was pool-only -- which is exactly the state `coverage_problems` exists to make visible.
    """
    x = torch.randn(M, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, device="cuda", dtype=torch.float32)
    out = layernorm_fwd(x, w, eps=1e-6)
    ref = torch.nn.functional.layer_norm(x.float(), (256,), w.float(), None, 1e-6)
    # The file's bf16 pair, not an invented absolute bound: at these magnitudes a fixed atol reads
    # as a failure for output rounding alone (measured 0.06 against a 3.8-magnitude element).
    assert_elementwise(out.float(), ref, tolerance_bound(ref, 1e-2, 1e-2), what="out")


# ══════════════════════════════════════════════════════════════════════════════════════════════ #
#  transpose=True -- LayerNormTransposeSm90: LayoutLeft in, row-major out
#
#  Requirements, all refused at the front door and pinned by the unsupported regions above:
#    * x MUST be LayoutLeft, stride (1, M). A row-major x is READ as if it were LayoutLeft.
#    * N is a multiple of the SMEM swizzle atom: 16 elements at bf16/fp16, 32 at fp32. Below the
#      floor the CUTLASS DSL backend corrupts its heap during lowering rather than raising.
#    * N <= 1024 -- one CTA stages a whole row.
#    * M is 16-byte aligned (M is the contiguous mode of the (N, M) TMA view).
#    * Power-of-two is NOT required for either M or N; alignment-multiple IS.
# ══════════════════════════════════════════════════════════════════════════════════════════════ #

#: ``(M, N, dtype)`` for the transposing correctness grid. A LIST of cells, not a cross product:
#: **M is in this variant's compile key** (the grid and the row guard are both derived from it), so
#: every cell is its own cold compile and a product would be hundreds of them. Each row is chosen
#: for what it probes, and the dtypes are spread across rows rather than multiplied through.
_TRANSPOSE_SHAPES = [
    # Power-of-two-N anchors, all three dtypes at the narrowest N.
    (4096, 128, torch.bfloat16),
    (4096, 128, torch.float16),
    (4096, 128, torch.float32),
    (64, 256, torch.bfloat16),
    (4096, 512, torch.float32),
    (64, 1024, torch.bfloat16),
    # Non-power-of-two M -- the realistic case, since M is a token count that varies at runtime.
    (384, 128, torch.bfloat16),
    (1000, 256, torch.float32),
    (696, 512, torch.bfloat16),
    (1512, 1024, torch.float32),
    # Non-power-of-two N -- D_hidden is any multiple of the swizzle atom. 96/192/384/768 clear the
    # fp32 floor (%32) as well as the 16-bit one, so both dtypes are reachable on these rows.
    (384, 96, torch.bfloat16),
    (384, 192, torch.float32),
    (696, 384, torch.bfloat16),
    (1000, 768, torch.float32),
    (696, 192, torch.bfloat16),
    (1000, 384, torch.float32),
    (1512, 768, torch.bfloat16),
    # 16-bit-only N: a multiple of 16 that is NOT a multiple of 32, i.e. below the fp32 floor.
    (1000, 80, torch.bfloat16),
    (1000, 80, torch.float16),
    (696, 80, torch.bfloat16),
    (1000, 160, torch.float16),
]


def _run_transpose(M, N, dtype, do_transpose, eps=1e-6, seed=0):
    """Run the transposing variant at one cell and return ``(out, reference)``.

    Args:
        M: Token extent. Must be 16-byte aligned at ``dtype`` or the front door refuses it.
        N: Feature extent. Must clear the swizzle-atom floor and be <= 1024.
        dtype: Element type of x and the output.
        do_transpose: ``"SMEM"`` / ``"shuffle"`` / ``"auto"``. ``"ldmatrix"`` raises.
        eps: Variance epsilon, passed to both the kernel and the reference.
        seed: RNG seed, so a failing cell is reproducible.

    Returns:
        ``(out, ref)`` -- the kernel's ``(M, N)`` row-major output and an fp32
        ``F.layer_norm`` reference of the SAME values. The input is built as
        ``randn(N, M).t()``, which is LayoutLeft in one allocation; ``x.float()`` then reads the
        identical values, so the reference is not computed from a second draw.
    """
    torch.manual_seed(seed)
    x = torch.randn(N, M, device="cuda", dtype=dtype).t()  # (M, N), stride (1, M) == LayoutLeft
    assert x.shape == (M, N) and x.stride() == (1, M), (x.shape, x.stride())
    weight = torch.randn(N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)
    out = layernorm_fwd(
        x, weight, bias, eps=eps, transpose=True, do_transpose=do_transpose, select="default"
    )
    ref = torch.nn.functional.layer_norm(x.float(), (N,), weight, bias, eps)
    return out, ref


@LAYERNORM.parametrize(
    "M",
    "N",
    "input_dtype",
    "do_transpose",
    cells=[(M, N, dt, mode) for (M, N, dt) in _TRANSPOSE_SHAPES for mode in ("SMEM", "shuffle")],
    because=(
        "M is in the TRANSPOSING variant's compile key -- unlike the row-major path, where it is a "
        "symbolic dim -- so every (M, N, dtype, mode) is its own cold compile and a cross product "
        "would be hundreds of them. The grid is therefore a list of cells, each chosen for what it "
        "probes (power-of-two anchor / non-pow2 M / non-pow2 N / 16-bit-only N), with the dtypes "
        "spread across rows. 'ldmatrix' and 'auto' are excluded: 'ldmatrix' is a declared "
        "unsupported region, and 'auto' resolves to one of the two swept here anyway, so it is "
        "covered by test_transpose_auto_matches_the_resolved_mode instead."
    ),
)
def test_transpose_forward(M, N, input_dtype, do_transpose):
    """LayerNorm over the STRIDED axis matches the fp32 reference, and the output is row-major.

    The output layout is asserted, not assumed: the whole point of this variant is the LayoutLeft ->
    row-major flip, so a kernel that computed the right values into a column-major buffer would be
    useless to its caller and every value-only assertion would still pass.

    The tolerance is this file's per-dtype ``atol``/``rtol`` pair, the same one every other forward
    test here uses. It replaces a relative L2 NORM ratio, which was the worst available shape for
    this check: a norm averages over M*N elements, so the strided-axis defects this variant exists
    to catch -- a mis-transposed tile, a dropped tail row -- move it by almost nothing. The reduction
    order does differ from ``F.layer_norm``'s, but that argues for the size of the per-element bound,
    not for pooling it away.
    """
    out, ref = _run_transpose(M, N, input_dtype, do_transpose)
    assert out.shape == (M, N) and out.dtype == input_dtype
    assert out.stride() == (N, 1), f"output must be row-major; got stride {out.stride()}"
    atol = rtol = 1e-2 if input_dtype is not torch.float32 else 1e-4
    assert_elementwise(
        out.float(),
        ref,
        tolerance_bound(ref, atol, rtol),
        what=f"M={M} N={N} {input_dtype} {do_transpose}",
    )


@LAYERNORM.parametrize(
    "M",
    "N",
    "input_dtype",
    cells=[
        (M, N, dt)
        for (M, N, dt) in _TRANSPOSE_SHAPES
        if (M, N) in {(4096, 128), (64, 256), (4096, 512), (64, 1024), (384, 96), (1000, 80)}
    ],
    because=(
        "bit-identity is a property of the two FRAGMENT-BUILD strategies, not of the shape, so the "
        "cells are the six that span the tile geometry (each N bucket, plus a non-pow2 and a "
        "16-bit-only N). Re-running the whole grid would double every compile in "
        "test_transpose_forward to re-prove the same equality."
    ),
)
def test_transpose_modes_are_bit_identical(M, N, input_dtype):
    """``do_transpose="SMEM"`` and ``"shuffle"`` produce BITWISE identical output.

    They differ only in where the reduction fragment is read from -- an ``sT`` round-trip tile
    versus a direct per-thread gather out of the swizzled ``sX``. The arithmetic that follows is
    the same instruction sequence on the same values, so this must be exact and not merely close.

    **Asserting equality rather than a tolerance is the point.** ``do_transpose`` is a perf knob
    (``"shuffle"`` is 7-15% faster at N<=256, ties above), and a tolerance would hide a knob that
    had quietly become a numerical choice. It is also what licenses the heuristic in
    ``_transpose_heuristic_config`` to pick between them on speed alone.
    """
    smem, _ = _run_transpose(M, N, input_dtype, "SMEM")
    shuffle, _ = _run_transpose(M, N, input_dtype, "shuffle")
    assert_bitwise(
        smem,
        shuffle,
        what=f"M={M} N={N} {input_dtype}: SMEM and shuffle disagree -- they are two ways to read the same staged tile, so max|diff| must be 0, got {(smem.float() - shuffle.float()).abs().max().item():.3e}",
    )


@LAYERNORM.parametrize(
    "M",
    "N",
    cells=[(M, N) for M in (262144, 1048576, 4194304, 9437184) for N in (128, 256, 384, 512)],
    because=(
        "the subject is the A2A-fused TriMul production grid specifically: D in {128,256,384,512} "
        "and M = N_token**2 / cp for N_token in {2048,4096,8192,12288} at cp=16. bfloat16 only, "
        "because these are the dtype the workflow runs and each cell already allocates 0.1-19 GB; "
        "the dtype axis is swept at small shapes in test_transpose_forward."
    ),
)
def test_transpose_production_token_counts(M, N):
    """The A2A-fused TriMul shapes compile and compute, at every declared token count.

    **M is in this variant's compile key**, so these are not a re-run of a smaller shape -- each is
    a distinct kernel, and the largest of them (M = 9437184) is 36 grid-widths past anything else
    in this file.

    The reference is computed on the first and last 4096 rows only. That is a memory decision, not
    a coverage one: an fp32 ``F.layer_norm`` over the full 9437184 x 512 tensor is 19 GB on top of
    the 19 GB the input and output already hold. The two slices bracket the grid, so a CTA-indexing
    error at either end still fails here, and every row is still COMPUTED.

    **OOM is a runtime skip, never a memory-estimate gate** (CLAUDE.md): a gate would have to guess
    the free memory of a GPU the rest of the suite is also using, and would silently stop testing
    the shape the day the guess drifted.
    """
    try:
        x = torch.randn(N, M, device="cuda", dtype=torch.bfloat16).t()  # LayoutLeft in one alloc
        weight = torch.randn(N, device="cuda", dtype=torch.float32)
        bias = torch.randn(N, device="cuda", dtype=torch.float32)
        out = layernorm_fwd(x, weight, bias, eps=1e-6, transpose=True)
    except torch.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        pytest.skip(f"M={M} N={N} does not fit on this device: {e}")
    assert out.shape == (M, N) and out.stride() == (N, 1)
    for lo, hi, where in ((0, 4096, "head"), (M - 4096, M, "tail")):
        ref = torch.nn.functional.layer_norm(x[lo:hi].float(), (N,), weight, bias, 1e-6)
        got = out[lo:hi].float()
        assert_elementwise(
            got,
            ref,
            tolerance_bound(ref, 1e-2, 1e-2),
            what=f"M={M} N={N} {where} rows [{lo}:{hi}]",
        )
    del x, out
    torch.cuda.empty_cache()


@LAYERNORM.parametrize(
    "N",
    "M",
    only={"N": [528, 560, 672, 784, 800, 864, 880, 1008], "M": [4096]},
    because=(
        "these are the eight feature extents whose auto-picked `subt` was NON-TILEABLE before the "
        "`_subt_is_tileable` fix, and seven of them are reachable from the shipped heuristic. They "
        "earn a test of their own because their old failure mode was a SIGABRT with no Python "
        "traceback, which no other test in this file can produce. One M, because the axis under "
        "test is the feature extent's tile choice."
    ),
)
def test_transpose_tileability_shapes_run_and_match_reference(N, M):
    """The feature extents that used to ABORT the process now run, and the values are right.

    Compiling is not correctness. Before the ``_subt_is_tileable`` fix, ``_auto_subt`` returned the
    largest divisor of N under the bucket base on the documented -- and wrong -- assumption that any
    divisor was valid, and ``cute.slice_`` then SIGABRT'd inside the swizzled composed layout: no
    exception, no traceback, the process simply died. Seven of these eight are reachable from the
    shipped size heuristic, so this was a production-path crash, not a corner.

    The margin is printed rather than only asserted: a bare bound tells you the cell passed but not
    whether it cleared by 6x or by 5%. Run with ``-s`` to see them.
    """
    out, ref = _run_transpose(M, N, torch.bfloat16, "auto")
    subt = _transpose_default_config(N, cutlass.BFloat16)[1]
    assert _subt_is_tileable(subt) and N % subt == 0, f"N={N} picked non-tileable subt={subt}"
    worst = assert_elementwise(
        out.float(), ref, tolerance_bound(ref, 1e-2, 1e-2), what=f"N={N} subt={subt}"
    )
    # The margin is printed rather than only asserted: a bare pass tells you the cell cleared but
    # not whether by 6x or by 5%. `assert_elementwise` returns the worst |err|/bound ratio, so the
    # margin is 1/worst. Run with `-s` to see them.
    print(f"REL N={N} subt={subt} worst_ratio={worst:.3e} margin={1.0 / max(worst, 1e-12):.1f}x")


@LAYERNORM.parametrize(
    "M",
    "N",
    "input_dtype",
    cells=[(4096, 128, torch.bfloat16), (4096, 1024, torch.bfloat16), (4096, 512, torch.float32)],
    because=(
        "the subject is the knob-RESOLUTION plumbing (heuristic / auto / autotune / freeze), which "
        "is shape-independent apart from the one N threshold it keys on. Three cells bracket that "
        "threshold: N=128 below it (resolves to 'shuffle'), N=1024 and N=512 above ('SMEM')."
    ),
)
def test_transpose_auto_matches_the_resolved_mode(M, N, input_dtype):
    """Every way of resolving ``do_transpose`` lands on a mode that computes the same bits.

    Four entries into the same kernel -- the size heuristic (the default), the explicit ``"auto"``
    rule, the measured sweep and the frozen config -- and the two modes they can resolve to are
    bit-identical, so **all four must agree exactly**. That is a stronger statement than "each is
    correct": it pins that the resolution layer cannot change the answer, only the speed, which is
    what licenses it to choose on timing alone.

    ``select="autotune"`` is included deliberately. It is a production-reachable path (CLAUDE.md),
    it compiles both candidates, and nothing else in this file would notice if the sweep started
    returning a config the fixed path cannot run.
    """
    torch.manual_seed(0)
    x = torch.randn(N, M, device="cuda", dtype=input_dtype).t()
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    heuristic = layernorm_fwd(x, w, b, eps=1e-6, transpose=True)  # select="heuristic" by default
    auto = layernorm_fwd(x, w, b, eps=1e-6, transpose=True, do_transpose="auto", select="default")
    tuned = layernorm_fwd(x, w, b, eps=1e-6, transpose=True, select="autotune")
    frozen = layernorm_transpose_freeze(x, w, b, eps=1e-6)
    assert frozen.config.get("do_transpose") in ("SMEM", "shuffle"), frozen.config
    for name, got in (("auto", auto), ("autotune", tuned), ("frozen", frozen(x, w, b))):
        assert_bitwise(
            heuristic,
            got,
            what=f"M={M} N={N} {input_dtype}: select={name!r} disagrees with the heuristic. The modes it can pick are bit-identical, so the resolution layer must not move a bit.",
        )


# ── the transposing variant's HOST-side tile choice: pure functions, no GPU ────────────────────
#: Feature extents that ABORTED the process before the `_subt_is_tileable` fix, with the `subt`
#: each used to pick. Reachable from the shipped default: the size heuristic routes D >= 512 here.
_CRASHED_D = {560: 56, 672: 56, 784: 56, 800: 50, 864: 54, 880: 55, 1008: 63}
#: Feature extents that worked before the fix. Their `subt` MUST be unchanged -- a different tile is
#: a different layout and a different speed, i.e. a regression dressed as a bug fix.
_WORKED_D = {512: 32, 640: 64, 768: 64, 896: 64, 1024: 64}


@pytest.mark.parametrize(
    "subt,expected",
    [
        (8, True),
        (16, True),
        (32, True),
        (48, True),
        (64, True),
        (80, True),
        (96, True),
        (128, True),
        # measured aborts: k = ceil(subt/8) odd and >= 3
        (24, False),
        (40, False),
        (56, False),
        (72, False),
        # measured aborts / quantisation: subt % 8 != 0
        (50, False),
        (54, False),
        (55, False),
        (63, False),
    ],
)
@matrix_exempt(
    "pins a PURE host-side predicate against layouts printed from a live MLIR trace; its axis is a "
    "candidate sub-tile edge, which is a tile parameter and not one of the kernel's shape axes"
)
def test_transpose_subt_tileable_matches_measured_layouts(subt, expected):
    """The tileability predicate matches the layouts that were PRINTED from a live MLIR trace.

    Every row is a measurement, not a derivation: for each ``subt`` the layout was built in a fresh
    process and the outcome recorded as OK / SIGABRT / ValueError. The two False groups have
    different causes -- ``k = ceil(subt/8)`` odd and >= 3 aborts inside ``cute.slice_``, while
    ``subt % 8 != 0`` quantises the tile so ``make_tiled_tma_atom`` rejects it -- and both are
    represented, because a predicate fitted to only one of them would pass half this list.
    """
    assert _subt_is_tileable(subt) is expected


@pytest.mark.parametrize("D", sorted(_WORKED_D))
@matrix_exempt(
    "asserts a PURE host-side tile choice is UNCHANGED for extents that already worked; two of the "
    "extents (640, 896) are outside the declared pool precisely because nothing needs to run them"
)
def test_transpose_working_shapes_keep_their_tile(D):
    """The tileability fix must not perturb a shape that already worked -- same subt, same speed.

    A tile change at these extents would be a silent perf regression: the kernel still computes the
    right answer, so no correctness test would notice.
    """
    _, subt, _ = _transpose_default_config(D, cutlass.BFloat16)
    assert subt == _WORKED_D[D], (
        f"D={D} tile changed {_WORKED_D[D]} -> {subt}; that is a perf change"
    )


@pytest.mark.parametrize("D", sorted(_CRASHED_D))
@matrix_exempt(
    "asserts the PURE host-side tile choice moved off the aborting value; the kernel actually "
    "RUNNING at these extents is test_transpose_tileability_shapes_run_and_match_reference"
)
def test_transpose_previously_crashing_shapes_pick_a_tileable_subt(D):
    """The extents that aborted now pick a tileable, dividing ``subt`` -- and not the old one."""
    _, subt, _ = _transpose_default_config(D, cutlass.BFloat16)
    assert _subt_is_tileable(subt), f"D={D} still picks non-tileable subt={subt}"
    assert D % subt == 0, f"D={D} subt={subt} is not a divisor"
    assert subt != _CRASHED_D[D], f"D={D} still picks the crashing subt={subt}"


@pytest.mark.parametrize("D", [528, 576, 704, 832, 960])
@matrix_exempt(
    "the discriminator extents for the PURE host-side tile choice; 576 is outside the declared "
    "pool because its role is to separate two candidate rules, not to be run"
)
def test_transpose_discriminator_shapes_pick_a_tileable_subt(D):
    """The extents that separate 'power of two' from 'atom multiple' all land on a tileable tile.

    D=528 (subt 48) is the one that distinguishes those two candidate rules; the 64x-odd extents
    were the ``subt``-vs-``D%128`` discriminators. A rule fitted to powers of two passes every
    other host-side test here and fails this one.
    """
    _, subt, _ = _transpose_default_config(D, cutlass.BFloat16)
    assert _subt_is_tileable(subt) and D % subt == 0


@matrix_exempt(
    "the FIRST PRINCIPLE as an executable assertion: it sweeps every 16-byte-aligned D from 8 to "
    "1024, which is a claim about the whole domain rather than about any pool of test values"
)
def test_transpose_no_supported_shape_is_narrowed():
    """A tileable divisor exists for EVERY 16-byte-aligned D -- the fix narrowed nothing.

    Guaranteed rather than hoped: the repo's alignment floor forces ``D % 8 == 0``, and ``subt=8``
    is tileable, so restricting the choice to tileable divisors cannot reject any extent the kernel
    accepted before. This is the assertion that makes that argument checkable -- if a future change
    to the bucket bases or the predicate narrowed the domain by even one extent, it fails here
    rather than as a refused shape in production.
    """
    for D in range(8, 1025, 8):
        _, subt, _ = _transpose_default_config(D, cutlass.BFloat16)
        assert _subt_is_tileable(subt), f"D={D} -> non-tileable subt={subt}"
        assert D % subt == 0, f"D={D} -> subt={subt} does not divide D"


@matrix_exempt(
    "pins the k=1 boundary of a PURE host-side predicate; its axis is the swizzle atom count k, "
    "not a shape the kernel is launched at"
)
def test_transpose_auto_subt_k1_boundary():
    """``k=1`` (subt=8) is legal while ``k=3`` (subt=24) is not, and the fallback reaches 8.

    A single atom has nothing to pair, which is why the boundary is at ``k=1`` and not at "multiple
    of 16" -- a rule stated that way would reject ``subt=8`` and, with it, every N whose only
    tileable divisor is 8.
    """
    assert _subt_is_tileable(8) and not _subt_is_tileable(24)
    assert _auto_subt(8, 64) == 8  # only 8 divides; must be accepted, not rejected
    assert _auto_subt(24, 64) == 8  # 24 itself is non-tileable -> fall back to 8


@pytest.mark.parametrize("N", [24, 40, 72])
@matrix_exempt(
    "sweeps feature extents BELOW the transposing swizzle-atom floor, which are by construction "
    "not supported extents and so are not in the declared pool; the matrix covers the same refusal "
    "at one representative N and this widens it to the three measured backend-crash cases"
)
def test_transpose_below_swizzle_floor_raises_cleanly(N):
    """A sub-floor N is refused with an API-level error, not a CUTLASS DSL heap corruption.

    24, 40 and 72 are the three bf16 extents measured to crash the backend: each is ``% 8`` but not
    ``% 16``, so the swizzle atom cannot cover a minor row. The failure was a heap corruption during
    MLIR lowering -- no exception, no frame naming the caller -- which is why the refusal has to
    happen at the front door and why it is asserted to be a ``ValueError`` and not an
    ``AssertionError`` (``python -O`` strips asserts, so an assert-based guard is not a guard).
    """
    x = _layout_left(torch.randn(512, N, device="cuda", dtype=torch.bfloat16))
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="swizzle atom"):
        layernorm_fwd(x, w, w, eps=1e-6, transpose=True)


@matrix_exempt(
    "asserts the front door refuses transpose-only knobs on the row-major path; the subject is the "
    "argument surface, not any shape the kernel is launched at"
)
def test_transpose_knobs_are_refused_without_transpose():
    """A transposing knob passed with ``transpose=False`` raises instead of being ignored.

    This is the one caller mistake on this API with no symptom otherwise: the row-major kernel runs,
    returns a correct answer for the layout it was given, and the knob simply does nothing -- so a
    caller who meant to select the transposing variant concludes the knob has no effect rather than
    that they forgot ``transpose=True``.
    """
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, device="cuda", dtype=torch.float32)
    for kwargs in (
        {"do_transpose": "shuffle"},
        {"subt": 64},
        {"blk_m": 16},
        {"threads_per_row": 32},
        {"select": "autotune"},
    ):
        with pytest.raises(ValueError, match="are knobs of the TRANSPOSING variant"):
            layernorm_fwd(x, w, eps=1e-6, **kwargs)
    # ... and the same call without them is fine, so the guard is not simply always-on.
    assert layernorm_fwd(x, w, eps=1e-6).shape == (64, 256)
