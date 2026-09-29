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

"""Tests for ``fold_cp_ops.kernels.gemm`` -- the host entry and the default epilogue on top of it.

Where ``tests/kernels/test_gemm_sm90.py`` tests the mainloop's geometry, this module tests what
``gemm()`` adds: the epilogue terms (alpha, beta, C, row/column bias, accumulate-in-place), the
ragged and gathered call forms, and the front-door validation that decides which of those
combinations are legal at all.

**Every check in ``gemm()`` raises, and none of them asserts.** That is a deliberate change from
upstream, where nine of them were ``assert`` statements: ``python -O`` strips asserts, and a
stripped stride check does not fail loudly -- it builds a kernel reading the wrong strides, which is
a wrong answer rather than an error. The declared unsupported regions below are what keeps them
raising.
"""

import pytest
import torch

from fold_cp_ops._internal.arch import UnsupportedArchError
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops.kernels.gemm import gemm, GemmDefaultSm90
from fold_cp_ops.testing.numerics import assert_elementwise, epilogue_error_bound
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    dtype_facets,
    front_door_raises,
    int_facets,
    matrix_exempt,
)

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(_SM != 9, reason=f"gemm() needs sm_90; this GPU is sm_{_SM}0")

_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)

# ── the declared test matrix for this kernel ───────────────────────────────────────────────────
# **Add a config HERE, not to one test.** tests/perf/test_benchmark_perf_gemm.py imports this very
# object, so the configs that are timed and the ones that are correctness-tested cannot drift.
GEMM = KernelMatrix(
    kernel="gemm",
    axes=(
        Axis(
            name="ab_dtype",
            domain=(
                "16-bit (bfloat16/float16) or 8-bit (float8_e4m3fn/float8_e5m2) A and B -- the "
                "widths the SM90 WGMMA atom has a form for. float32 is in the pool because it is a "
                "KEY of torch2cute_dtype_map (D and the biases may be fp32) and so reaches the "
                "entry, where it must be refused as an OPERAND. Anything outside the map is "
                "refused earlier, by name"
            ),
            values=(
                torch.bfloat16,
                torch.float16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
                torch.float32,
            ),
            facets=dtype_facets(
                (
                    torch.bfloat16,
                    torch.float16,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                    torch.float32,
                )
            ),
        ),
        Axis(
            name="a_major",
            domain="'k' or 'm'; fp8 operands must be 'k' because Hopper has no mn-major fp8 atom",
            values=("k", "m"),
            facets={"k_major": lambda v: v == "k", "m_major": lambda v: v == "m"},
        ),
        Axis(
            name="persistent",
            domain=(
                "bool; True launches one resident wave that loops over work tiles, False one CTA "
                "per tile. Required by pingpong"
            ),
            values=(False, True),
            facets={"persistent": lambda v: v, "one_shot": lambda v: not v},
        ),
        Axis(
            name="add_to_output",
            domain="bool; accumulate into D rather than overwrite it",
            values=(False, True),
            facets={"accumulate": lambda v: v, "overwrite": lambda v: not v},
        ),
        Axis(
            name="alpha_mode",
            domain=(
                "'one' (folded away entirely), 'scalar' (a host float baked into the kernel), or "
                "'tensor' (a device pointer read at launch). Three distinct compiled kernels, not "
                "one kernel with three inputs"
            ),
            values=("one", "scalar", "tensor"),
            facets={
                "absent": lambda v: v == "one",
                "compile_time": lambda v: v == "scalar",
                "runtime": lambda v: v == "tensor",
            },
        ),
        Axis(
            name="beta_mode",
            domain="same three modes as alpha, applied to the optional C addend",
            values=("one", "scalar", "tensor"),
            facets={
                "absent": lambda v: v == "one",
                "compile_time": lambda v: v == "scalar",
                "runtime": lambda v: v == "tensor",
            },
        ),
        Axis(
            name="bias",
            domain=(
                "'none' / 'row' ((l, n), broadcast down rows) / 'col' ((l, m), broadcast across "
                "columns) / 'both'. Each is a separate epilogue op, so each is a separate kernel"
            ),
            values=("none", "row", "col", "both"),
            facets={
                "no_bias": lambda v: v == "none",
                "has_row": lambda v: v in ("row", "both"),
                "has_col": lambda v: v in ("col", "both"),
            },
        ),
        Axis(
            name="layout",
            domain=(
                "how the operands are laid out in memory, independent of their SHAPE. This axis "
                "exists because the 16-byte floor is on the row PITCH and the base ADDRESS, never "
                "on an extent -- and a contiguous tensor makes the pitch equal the extent, which "
                "hides the distinction. 'padded_pitch' is what makes an odd or sub-floor extent "
                "legal; 'tight_off_floor' and 'unaligned_base' are the two ways to break it"
            ),
            values=("contiguous", "padded_pitch", "tight_off_floor", "unaligned_base"),
            facets={
                "legal": lambda v: v in ("contiguous", "padded_pitch"),
                "padded": lambda v: v == "padded_pitch",
                "breaks_pitch": lambda v: v == "tight_off_floor",
                "breaks_base": lambda v: v == "unaligned_base",
            },
        ),
        Axis(
            name="M",
            domain=(
                "any positive int as an extent -- M is tiled and predicated, so 1, 7, 63, 301 and "
                "1001 all compute. It picks up the 16-byte floor ONLY when D is m-major, where M "
                "becomes the row pitch; that is a property of the layout, not of M"
            ),
            values=(1, 7, 63, 128, 301, 1000, 1001, 2048, 4096, 8192, 12288, 1000000),
            facets={
                **int_facets(tile=128, big=1000, small=64),
                # The BACK-half einsum's own token ladder. **This kernel takes the BACK law, not
                # the FRONT one**: op 7 is ``out[i,j] = sum_k a[i,k] b[j,k]`` with
                # ``M == N == K == N_token``, so the values that belong here are the N_token series
                # itself -- NOT the projection kernels' ``M = N_token**2 / cp``. Putting the front's
                # M pool here would decouple K from N and measure an O(N^2) thin-K matmul in place
                # of the O(N^3) einsum, which is the exact benchmark-invalidating substitution
                # CLAUDE.md's `K = N_token` rule exists to forbid.
                "workflow_N_token": lambda v: v in (2048, 4096, 8192, 12288),
            },
        ),
        Axis(
            name="N",
            domain=(
                "any positive int, as an EXTENT -- MEASURED: N = 1, 3, 7, 201 and 257 all compute "
                "correctly at bf16. The 16-byte floor lands on the row PITCH, not on N: "
                "torch.empty(l, m, 208)[:, :, :201] works and torch.empty(l, m, 201) does not, and "
                "the difference between them is the pitch. A CONTIGUOUS D conflates the two, which "
                "is why the everyday statement 'N must be a multiple of 8' is true and misleading"
            ),
            values=(1, 3, 7, 8, 64, 128, 200, 201, 256, 257, 384, 512, 768, 1000, 1024, 2048),
            facets={
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                "tile_aligned": lambda v: v % 128 == 0,
                "fp8_aligned": lambda v: v % 16 == 0,
                "large": lambda v: v >= 1000,
                "trimul_postact": lambda v: v in (256, 512, 768, 1024),
            },
        ),
        Axis(
            name="K",
            domain=(
                "any positive int, as an EXTENT -- MEASURED: K = 1, 3, 7 and 129 all compute "
                "correctly at bf16 with A/B's pitch padded to the floor. Same pitch-vs-extent "
                "split as N"
            ),
            values=(1, 3, 7, 8, 64, 128, 129, 192, 256, 384, 512, 1000, 2048),
            facets={
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                "sub_tile": lambda v: v < 64,
                "tile_aligned": lambda v: v % 64 == 0,
                "large": lambda v: v >= 1000,
                "trimul_feature_dim": lambda v: v in (128, 256, 384),
                # 512 completes the four TriMul feature widths; it was the one absent from this
                # pool while the other three were named by `trimul_feature_dim`, which is how a
                # width can rot out of coverage without the facet noticing.
                "workflow_D": lambda v: v in (128, 256, 384, 512),
            },
        ),
        Axis(
            name="L",
            domain="any positive int; the batch count",
            values=(1, 2, 3, 7, 8, 16, 32, 48),
            facets={
                "single_batch": lambda v: v == 1,
                "multi_batch": lambda v: v > 1,
                "odd_batch": lambda v: v % 2 == 1,
                # L = Dloc * B for the A2A-fused TriMul back einsum, where Dloc = D / cp and
                # B is the sequence batch. The profiled production grid
                # (profiling/distributed/trimul/dlc_composite_k_dtensor_cp_le16) runs D in
                # {128, 256, 384} at cp up to 16 with B in {1, 2}, so Dloc reaches 8 and the
                # reachable L values are 8, 16, 32 and 48. 8 was MISSING from this pool -- it is
                # D=128 at cp=16, which the grid covers -- and the gap was invisible because the
                # pool jumped 7 -> 16 and 7 looks like a batch test.
                "trimul_per_device": lambda v: v in (8, 16, 32, 48),
            },
        ),
        Axis(
            name="arg_fault",
            domain=(
                "a malformed TENSOR ARGUMENT, or 'none'. The values name a CALLER MISTAKE rather "
                "than a kernel configuration, and every non-'none' one must be refused at the front "
                "door with an API-level error. Both a dtype and an extent are in the pool because a "
                "caller can get either wrong on any argument, and which arguments the kernel feeds "
                "to a TMA versus reads with a broadcast load is not visible in its signature. "
                "'operand_extent' and 'C_extent' are the PARTNER faults: an extent that is legal "
                "on its own tensor and disagrees with the tensor it must match, which nothing "
                "compared before -- they reached the TVM-FFI ABI check, which names an argument "
                "index and a traced symbol rather than the argument the caller passed"
            ),
            values=(
                "none",
                "rowvec_dtype",
                "rowvec_extent",
                "operand_extent",
                "C_extent",
            ),
            facets={
                "well_formed": lambda v: v == "none",
                "bad_dtype": lambda v: v.endswith("_dtype"),
                "bad_extent": lambda v: v.endswith("_extent"),
            },
        ),
    ),
    # ORDER MIRRORS `gemm()`'s CHECK ORDER, and that is load-bearing rather than tidy:
    # `parametrize_unsupported` stops at the FIRST matching region, so a combo in several regions is
    # tested against whichever is listed first. List them out of order and the test asserts a
    # message the kernel does not produce -- which is how the layout regions once failed 433 cells
    # by sitting ahead of the checks that actually fire first. The sequence below is: operand
    # WIDTH, operand LAYOUT (pitch, then base), fp8 major. The width check sits inside
    # `describe_operands` -- fused with the layout pass for launch latency -- so it fires before
    # both layout checks even though it reads as a dtype check. Move it and this tuple moves with
    # it.
    # A contraction plus an epilogue. The dtype pool reaches both fp8 encodings.
    computes=("contraction", "fp8_operands"),
    unsupported=(
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                arg_fault == "operand_extent"
                and ab_dtype == torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
            ),
            raises=ValueError,
            match=r"must contract over the same extent",
            reason=(
                "every operand was validated ALONE and none was compared to the others, so A and B "
                "disagreeing on K passed each individual check and reached the TVM-FFI ABI check "
                "as `Mismatched mB.shape[1] on argument #1`, naming a traced symbol and an FFI "
                "slot rather than the argument the caller passed. Both tensors are individually "
                "well formed here, which is the point: the fault exists only in the relationship "
                "between them, and that is the class of check this entry had none of"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                arg_fault == "C_extent"
                and ab_dtype == torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
            ),
            raises=ValueError,
            match=r"C must be D's shape",
            reason=(
                "C is added elementwise and is never broadcast, so a C that is not D's shape reads "
                "past its own end on the trailing tiles -- silently, since a mapped page follows. "
                "The same guard covers D's own extent, which until now was validated only as a "
                "SIDE EFFECT of `rowvec_bias` being present (that check compares against "
                "D.shape[-1]), so a mis-sized D went unnamed whenever the optional arguments were "
                "absent. A pass contingent on an unrelated argument reads as coverage and is not"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype: ab_dtype == torch.float32,
            raises=ValueError,
            match=r"unsupported operand dtype torch.float32 for A",
            reason=(
                "the SM90 WGMMA atom has 16-bit and 8-bit forms only; there is no fp32 MMA on "
                "Hopper. fp32 is nonetheless a key of torch2cute_dtype_map -- D and the broadcast "
                "biases legitimately are fp32 -- so it passes the dtype-membership check and used "
                "to be refused 59 ms later from inside tracing, by GemmSm90.__call__. It is now "
                "refused at the host entry in 39 us, which is what makes this region cheap enough "
                "to sweep"
            ),
        ),
        Unsupported(
            where=lambda layout: layout == "tight_off_floor",
            raises=ValueError,
            match=r"violates the 16-byte alignment floor",
            reason=(
                "a CONTIGUOUS tensor whose trailing extent is off the floor has a row pitch off "
                "the floor too, and a TMA descriptor's stride must be a multiple of 16 bytes. The "
                "extent itself is fine -- the SAME extent with 'padded_pitch' computes correctly, "
                "which is exactly what test_gemm_extent_is_free_when_the_pitch_is_padded asserts. "
                "Declaring both sides is the only way the pitch-vs-extent distinction is stated "
                "rather than assumed"
            ),
        ),
        Unsupported(
            where=lambda layout: layout == "unaligned_base",
            raises=ValueError,
            match=r"does not start on a 16-byte boundary",
            reason=(
                "a TMA descriptor's global address must be 16-byte aligned, which a slice at a "
                "non-multiple offset breaks -- D[..., 1:] on a 16-bit tensor starts 2 bytes in. "
                "Without the front-door check this surfaced as `Misaligned Tensor data on argument "
                "#2` from the FFI, which names neither the operand the caller passed nor the fix"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, a_major: ab_dtype in _FP8 and a_major != "k",
            raises=ValueError,
            match=r"must be k-major",
            reason=(
                "Hopper's 8-bit WGMMA atom exists only in the k-major form. Refused here rather "
                "than left to fail as `OpError: only support f16/bf16 with mn-major` from inside "
                "partition_fragment_ABC, which names neither the operand nor the fix"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                ab_dtype == torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
                and arg_fault == "rowvec_dtype"
            ),
            raises=ValueError,
            match=r"unsupported dtype for rowvec_bias",
            reason=(
                "a broadcast vector's dtype is only a LOAD WIDTH, so nothing downstream refuses it -- before the front door checked it the only thing that failed was the dict lookup building the compile key, a bare KeyError naming a torch dtype and no argument"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                ab_dtype == torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
                and arg_fault == "rowvec_extent"
            ),
            raises=ValueError,
            match=r"rowvec_bias must be",
            reason=(
                "the row vector is broadcast along M with stride 0, so one shorter than N biases the tail columns with whatever follows it in memory"
            ),
        ),
    ),
)


def _pad_to_floor(extent, dtype):
    """Round an extent up to the row pitch a TMA descriptor requires for ``dtype``.

    Args:
        extent: The logical extent.
        dtype: Element type; the floor is ``128 // width`` elements -- 8 at 16-bit, 16 at fp8.

    Returns:
        The smallest multiple of the floor that is >= ``extent``; ``extent`` itself when it already
        conforms, so callers may apply it unconditionally.
    """
    from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map

    div = 128 // torch2cute_dtype_map[dtype].width
    return (extent + div - 1) // div * div


def _gen(shape, dtype, device="cuda", layout="contiguous"):
    """Allocate one operand under a declared memory layout, scaled so an fp8 cast does not saturate.

    **The layout is applied to the ALLOCATION, not to the finished view.** That distinction is the
    whole reason this is one function rather than a generator plus a post-processing pass: a
    post-processing pass copies into a fresh contiguous buffer, which silently converts an m-major
    operand into a k-major one. When it did, 48 cells stopped raising -- not because the layout was
    wrong but because the *major* had changed underneath the region that was being tested.

    Args:
        shape: The logical shape. Only the LAST axis is padded or offset; a caller that wants the
            other axis affected transposes the RESULT, which is what preserves the major.
        dtype: Any dtype in the ``ab_dtype`` pool. fp8 values come from ``randn() / 4``, inside
            e4m3's range, so a comparison measures the kernel rather than the cast.
        device: CUDA device.
        layout: One of the ``layout`` axis's values.

            * ``"contiguous"`` / ``"tight_off_floor"`` -- a plain allocation. They differ only in
              the caller's *intent*: under ``tight_off_floor`` the caller passes an extent off the
              16-byte floor, so the pitch is off the floor too and the kernel must refuse.
            * ``"padded_pitch"`` -- the trailing axis is allocated up to the floor and sliced back,
              giving the same extent with a conforming pitch. This is what makes an odd extent legal.
            * ``"unaligned_base"`` -- conforming pitch, but the view starts one element into the
              allocation, so the base address is off a 16-byte boundary.

    Returns:
        A tensor of exactly ``shape``.
    """
    want = shape[-1]
    if layout == "padded_pitch":
        alloc = shape[:-1] + (_pad_to_floor(want, dtype),)
        lo = 0
    elif layout == "unaligned_base":
        alloc = shape[:-1] + (_pad_to_floor(want + 1, dtype),)
        lo = 1
    else:
        alloc, lo = shape, 0
    if dtype in _FP8:
        t = (torch.randn(*alloc, device=device) / 4).to(dtype)
    else:
        t = torch.randn(*alloc, device=device, dtype=dtype)
    return t[..., lo : lo + want]


def _zeros(shape, dtype, device="cuda", layout="contiguous"):
    """A zeroed output view under the same layout rules as :func:`_gen`.

    D carries the layout as well as A and B, and for the two illegal layouts it is usually D that
    trips them -- N lives on D's trailing axis, so ``tight_off_floor`` with an odd N is a D-side
    violation. Zeroed rather than uninitialized because ``add_to_output`` reads D back.

    Args:
        shape: The logical shape.
        dtype: The output element type.
        device: CUDA device.
        layout: One of the ``layout`` axis's values; see :func:`_gen`.

    Returns:
        A zeroed tensor of exactly ``shape``.
    """
    want = shape[-1]
    if layout == "padded_pitch":
        alloc, lo = shape[:-1] + (_pad_to_floor(want, dtype),), 0
    elif layout == "unaligned_base":
        alloc, lo = shape[:-1] + (_pad_to_floor(want + 1, dtype),), 1
    else:
        alloc, lo = shape, 0
    return torch.zeros(*alloc, device=device, dtype=dtype)[..., lo : lo + want]


def call_gemm(
    ab_dtype=torch.bfloat16,
    a_major="k",
    persistent=True,
    add_to_output=False,
    alpha_mode="one",
    beta_mode="one",
    bias="none",
    layout="contiguous",
    M=256,
    N=256,
    K=256,
    L=2,
    tile_M=128,
    tile_N=128,
    cluster_N=1,
    device="cuda",
):
    """Build operands for one point of the matrix, call ``gemm()``, and return ``(D, reference)``.

    One helper rather than one per test because the axes interact: ``layout`` changes how every
    operand is allocated, ``a_major`` changes A's stride order, ``beta_mode`` decides whether a C
    addend exists at all, and the reference has to follow each of those. Splitting that across
    tests is how the reference and the call drift apart.

    Args:
        ab_dtype: Element type of A and B. fp8 forces a bf16 output, since an fp8 D would measure
            the store's saturation instead of the MMA.
        a_major: ``"k"`` for a contiguous A, ``"m"`` for a genuine transposed view.
        persistent: Forwarded to ``gemm()``.
        add_to_output: Accumulate into a pre-filled D rather than overwriting it. The reference
            adds the same initial values.
        alpha_mode: ``"one"`` / ``"scalar"`` / ``"tensor"``, matching the axis.
        beta_mode: Same, and it also decides whether a C addend is passed at all -- ``"one"`` means
            no C, so the beta term is compiled away rather than multiplied by one.
        bias: ``"none"`` / ``"row"`` / ``"col"`` / ``"both"``.
        layout: One of the ``layout`` axis's values, applied to A, B and D alike. Under
            ``"tight_off_floor"`` the caller is expected to pass an N or K that is off the floor --
            the layout name describes the *intent*, and a conforming extent under it is simply a
            contiguous tensor that will not raise.
        M: Rows. Free -- M is tiled and predicated, and picks up the 16-byte floor only when D
            is m-major, where M becomes the row pitch.
        N: Columns. The floor lands on D's row PITCH; under ``layout="padded_pitch"`` N itself is
            free.
        K: Contraction extent, same pitch-vs-extent split as N.
        L: Batch count.
        tile_M: CTA tile M.
        tile_N: CTA tile N.
        cluster_N: Cluster extent along N. Power of two, and ``cluster_M * cluster_N <= 8``.
        device: CUDA device.

    Returns:
        ``(D, ref)`` -- the kernel's output and an fp32 torch reference of the same shape.

    Raises:
        ValueError: Propagated from ``gemm()``'s front door, which is what the unsupported-region
            test relies on.
    """
    out_dtype = torch.bfloat16 if ab_dtype in _FP8 else ab_dtype
    # `a_major` builds A as a genuine transposed VIEW rather than a re-allocation, so the m-major
    # case really does hand the kernel a non-contiguous last axis. Copying into a fresh contiguous
    # buffer instead would silently turn every m-major cell back into a k-major one.
    A = (
        _gen((L, M, K), ab_dtype, layout=layout)
        if a_major == "k"
        else _gen((L, K, M), ab_dtype, layout=layout).transpose(-1, -2)
    )
    B = _gen((L, N, K), ab_dtype, layout=layout)
    D = _zeros((L, M, N), out_dtype, device, layout)
    D0 = D.clone() if add_to_output else None
    C = _gen(tuple(D.shape), out_dtype) if beta_mode != "one" else None
    alpha = {"one": 1.0, "scalar": 2.5}.get(alpha_mode) or torch.tensor(
        [2.5], device=device, dtype=torch.float32
    )
    beta = {"one": 1.0, "scalar": -0.5}.get(beta_mode) or torch.tensor(
        [-0.5], device=device, dtype=torch.float32
    )
    rv = _gen((L, N), torch.float32) if bias in ("row", "both") else None
    cv = _gen((L, D.shape[-2]), torch.float32) if bias in ("col", "both") else None
    gemm(
        A,
        B,
        D,
        C,
        None,
        tile_M,
        tile_N,
        1,
        cluster_N,
        persistent=persistent,
        rowvec_bias=rv,
        colvec_bias=cv,
        alpha=alpha,
        beta=beta,
        add_to_output=add_to_output,
    )

    a_scale = alpha.item() if torch.is_tensor(alpha) else alpha
    b_scale = beta.item() if torch.is_tensor(beta) else beta
    ref = a_scale * (A.float() @ B.float().transpose(-1, -2))
    if C is not None:
        ref = ref + b_scale * C.float()
    if rv is not None:
        ref = ref + rv.unsqueeze(-2)
    if cv is not None:
        ref = ref + cv.unsqueeze(-1)
    if D0 is not None:
        ref = ref + D0.float()
    # The per-element bound is built HERE because this is the only place that still holds A, B and
    # the alpha that was applied. A relative bound computed later from `ref` alone is the wrong
    # model: an output element whose K terms cancel to near zero has a legitimately large RELATIVE
    # error and a perfectly normal ABSOLUTE one, and only sum_k|a*b| distinguishes the two.
    widen = 8.0 if ab_dtype in _FP8 else 1.0
    bound = epilogue_error_bound(A, B, ref, D.dtype, alpha=a_scale) * widen
    return D, ref, bound


def assert_close(D, ref, bound):
    """Compare against the exact reference **element by element**, against a per-element bound.

    A thin alias for :func:`assert_elementwise` so the call sites read the way they used to. What
    changed is the third argument: it was the operand dtype, from which one scalar tolerance was
    derived for the whole tensor, and it is now the tensor of per-element allowances that
    ``call_gemm`` built from the operands.

    Args:
        D: The kernel's output.
        ref: The exact fp64 reference for the whole epilogue -- alpha, beta and every bias applied.
        bound: Per-element allowance, shaped like ``ref``. Produced by ``call_gemm``; do not
            substitute a scalar, which would discard the point of the change.

    Returns:
        None.

    Raises:
        AssertionError: On a non-finite output or any element outside its own bound, reported with
            the violating count and the three worst coordinates, values, bounds and ratios.
    """
    assert_elementwise(D, ref, bound, what="gemm")


# ── the epilogue: every term, and every mode of every term ────────────────────────────────────
@requires_sm90
@GEMM.parametrize("alpha_mode", "beta_mode")
def test_gemm_alpha_beta_modes(alpha_mode, beta_mode):
    """alpha and beta each compile away, bake in, or read a pointer -- independently.

    The full 3 x 3 product is swept, with no ``because=`` to write, because the three modes are
    three *different compiled kernels* rather than three values of one input: mode 0 folds the
    multiply out of the epilogue, mode 1 bakes a constant in, mode 2 emits a load. A mode that
    stopped being folded away would show up nowhere else.

    ``beta_mode="one"`` also means *no C at all*: the term is absent from the kernel rather than
    multiplied by one, which is the distinction the cache key encodes.
    """
    torch.manual_seed(0)
    D, ref, bound = call_gemm(alpha_mode=alpha_mode, beta_mode=beta_mode)
    assert_close(D, ref, bound)


@requires_sm90
@GEMM.parametrize("bias", "add_to_output")
def test_gemm_bias_and_accumulate(bias, add_to_output):
    """Row bias, column bias, both, and accumulate-in-place, in every combination.

    The full 4 x 2 product, again with nothing to restrict: each bias value is a different epilogue
    op list and therefore a different kernel, and ``add_to_output`` changes the store from a write
    to a read-modify-write. Shape and dtype are held fixed so a failure names the epilogue.
    """
    torch.manual_seed(0)
    D, ref, bound = call_gemm(bias=bias, add_to_output=add_to_output)
    assert_close(D, ref, bound)


@requires_sm90
@GEMM.parametrize(
    "ab_dtype",
    "a_major",
    drop={"a_major": ["m"], "ab_dtype": [torch.float32]},
    because=(
        "A stays k-major because fp8 requires it -- (fp8, m-major) is a declared unsupported "
        "region, and test_gemm_major_modes covers the 16-bit m-major path. float32 is dropped "
        "because it is itself a declared region: legal as a D and bias type, not as an OPERAND "
        "type, and a supported-path test cannot assert a correct answer for a combo the matrix "
        "says must raise."
    ),
)
def test_gemm_dtypes(ab_dtype, a_major):
    """All four operand types run through the host entry, fp8 included.

    fp8 reaching ``gemm()`` is new: ``torch2cute_dtype_map`` had no fp8 entries, so a call died on
    ``KeyError: torch.float8_e4m3fn`` while ``GemmSm90`` claimed to support it. N and K are 16-byte
    aligned at 8 bits here (256 = 16 x 16), which the 16-bit-only cells elsewhere need not be.
    """
    torch.manual_seed(0)
    D, ref, bound = call_gemm(ab_dtype=ab_dtype, a_major=a_major)
    assert_close(D, ref, bound)


@requires_sm90
@GEMM.parametrize(
    "a_major",
    "ab_dtype",
    only={"ab_dtype": [torch.bfloat16, torch.float16]},
    because=(
        "major modes are a 16-bit-only question: every non-k-major fp8 operand is a declared "
        "unsupported region, so sweeping fp8 here would parametrize cells the matrix refuses."
    ),
)
def test_gemm_major_modes(a_major, ab_dtype):
    """A may be contiguous along either axis at 16 bits, and the result is the same."""
    torch.manual_seed(0)
    D, ref, bound = call_gemm(ab_dtype=ab_dtype, a_major=a_major)
    assert_close(D, ref, bound)


# ── shapes: the 16-byte floor is the only constraint ──────────────────────────────────────────
@requires_sm90
@GEMM.parametrize(
    "M",
    "N",
    "K",
    "L",
    cells=[
        (1, 8, 8, 1),  # the degenerate floor
        (7, 64, 64, 1),  # M far below tile_M
        (63, 128, 192, 2),  # M just under a tile
        (301, 200, 256, 3),  # off-grid M and N, odd batch
        (1001, 384, 1000, 1),  # odd M, off-grid K
        (4096, 2048, 2048, 1),  # large enough for the persistent grid to loop
        (128, 1000, 8, 7),  # off-grid N with the minimum K and the largest batch
        # --- the A2A-fused TriMul's own shapes, through the full default epilogue ---
        (1000, 1000, 1000, 16),  # BACK: the square einsum, M == N == K == N_token, L = D/cp
        (4096, 512, 128, 1),  # FRONT: K = D = 128, the O(N^2) projection (exempt from K = N)
        (4096, 1024, 256, 1),  # FRONT: K = D = 256
    ],
    because=(
        "shapes are a list, not a product: the pools cross to thousands of cells, each a full "
        "recompile. The first 7 span the axes independently -- M below/at/over a tile and odd, K "
        "below/at/over one k-tile, N off-grid and large, L single/even/odd -- inside the 16-byte "
        "floor. The last 3 are the TriMul regimes the generic cells do not reach: the square back "
        "einsum, and the thin-K front projection at two of its three feature dims. Each is timed "
        "in tests/perf/test_benchmark_perf_gemm.py at its production size; the correctness cells "
        "hold M down because the fp32 reference is what would not fit, not the kernel."
    ),
)
def test_gemm_shapes(M, N, K, L):
    """The declared shape freedom is real through the host entry, epilogue included.

    ``bias="row"`` and not ``"both"``: a **column** bias is shaped ``(l, m)``, which makes M its row
    pitch and so imposes a 4-element alignment on an extent that is otherwise entirely free. That
    interaction is real and worth knowing, but it belongs to the bias and not to the shape sweep --
    ``test_colvec_bias_imposes_an_alignment_on_M`` pins it separately, and this test would
    otherwise be unable to carry M = 63 or 301 at all.
    """
    torch.manual_seed(0)
    D, ref, bound = call_gemm(M=M, N=N, K=K, L=L, bias="row", alpha_mode="scalar")
    assert_close(D, ref, bound)


@requires_sm90
@matrix_exempt("an interaction between one epilogue term and one extent; nothing else varies")
def test_colvec_bias_imposes_an_alignment_on_M():
    """Passing a column bias makes M the bias's row pitch, so M picks up a 4-element floor.

    Not a second shape constraint on the GEMM: M is free at every cell of ``test_gemm_shapes``. It
    is a constraint on the ``(l, m)`` bias tensor, which the epilogue loads with a 4-element
    vectorized copy. Worth a test because the failure otherwise reads ``Invalid
    epilogue_args[3].strides[0] ... expected to be divisible by 4``, which names neither the
    argument the caller passed nor the extent to pad.
    """
    torch.manual_seed(0)
    with pytest.raises(ValueError, match=r"colvec_bias has stride"):
        call_gemm(M=301, N=256, K=256, L=2, bias="col")
    D, ref, bound = call_gemm(M=304, N=256, K=256, L=2, bias="col")  # the next multiple of 4
    assert_close(D, ref, bound)


# ── the declared unsupported regions ──────────────────────────────────────────────────────────
@requires_sm90
@GEMM.parametrize_unsupported(
    "ab_dtype",
    "a_major",
    "layout",
    "arg_fault",
)
def test_gemm_unsupported_combos_raise(
    arg_fault,
    ab_dtype,
    a_major,
    layout,
    expected_error,
    expected_match,
):
    """Every declared unsupported combination is refused by ``gemm()`` itself, with a usable message.

    All of these raise before any kernel is compiled -- including the fp32-operand and layout
    regions, which used to cost 59 ms and an FFI round-trip respectively -- so the sweep stays cheap
    despite its size. The operands are deliberately tiny: nothing here computes, and the allocation
    was the dominant per-cell cost.

    ``layout`` is in the sweep because two regions read it, and ``parametrize_unsupported`` refuses
    to run without every region's axes: a sweep that silently skipped them would report coverage it
    does not have. The N passed under ``tight_off_floor`` is 201 -- an extent that is legal with a
    padded pitch and illegal contiguous, which is the entire distinction the region encodes.

    The assertion is on the raise and not on an xfail: an xfail is satisfied by a kernel that
    quietly accepts the combination and returns a wrong answer, which is the failure this matrix
    exists to prevent, whereas ``pytest.raises`` fails that kernel with "DID NOT RAISE".
    """
    torch.manual_seed(0)
    N = 201 if layout == "tight_off_floor" else 64
    # One malformed ARGUMENT, built here rather than in the matrix: a fault is not a configuration,
    # so it does not belong to the cell the other axes describe.
    # Applied ONLY on the cell the fault regions claim. Regions match first-wins, so on a bad-dtype
    # or bad-layout cell that region owns the expectation, and malforming the bias there would make
    # the kernel raise about the bias while the test asserted the other message.
    faultable = ab_dtype is torch.bfloat16 and a_major == "k" and layout == "contiguous"
    if faultable and arg_fault != "none":
        # `call_gemm` builds its bias internally, so a MALFORMED one has to go through gemm()
        # directly. Minimal well-formed operands, exactly one bad argument.
        A = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, 64, device="cuda", dtype=torch.bfloat16)
        D = torch.empty(64, N, device="cuda", dtype=torch.bfloat16)
        C, rv = None, None
        if arg_fault == "rowvec_dtype":
            rv = torch.randn(1, N, device="cuda", dtype=torch.float64)
        elif arg_fault == "rowvec_extent":
            rv = torch.randn(1, N + 1, device="cuda", dtype=torch.float32)
        elif arg_fault == "operand_extent":
            # B's contraction extent disagrees with A's. Both tensors are individually well formed,
            # which is the whole point: the fault only exists in the relationship between them.
            B = torch.randn(N, 72, device="cuda", dtype=torch.bfloat16)
        elif arg_fault == "C_extent":
            C = torch.randn(64, N + 8, device="cuda", dtype=torch.bfloat16)
        with front_door_raises(expected_error, expected_match):
            gemm(A, B, D, C, None, tile_M=64, tile_N=64, rowvec_bias=rv)
        return
    with front_door_raises(expected_error, expected_match):
        call_gemm(
            ab_dtype=ab_dtype,
            a_major=a_major,
            layout=layout,
            M=64,
            N=N,
            K=64,
            L=1,
        )


@requires_sm90
@GEMM.parametrize(
    "N",
    "layout",
    only={"N": [1, 3, 7, 201, 257], "layout": ["padded_pitch"]},
    because=(
        "the subject is exactly the extents that are NOT multiples of the 16-byte floor, under the "
        "one layout that makes them legal. The conforming extents are covered by test_gemm_shapes; "
        "the other three layouts are either that same test's case or a declared region."
    ),
)
def test_gemm_extent_is_free_when_the_pitch_is_padded(N, layout):
    """Odd and sub-floor N compute correctly once the row pitch is padded to the floor.

    The supported half of the pitch-vs-extent distinction, and the reason the ``tight_off_floor``
    region is a statement about the *pitch* rather than about N. Same N, two layouts, two outcomes:
    ``torch.empty(l, m, 201)`` is refused and ``torch.empty(l, m, 208)[:, :, :201]`` is correct.

    N = 1 and 3 are worth their cells beyond being odd -- they are two and six bytes, well under the
    16-byte figure CLAUDE.md names as the repo-wide floor, which is precisely the claim being
    corrected: that figure bounds the pitch, not the extent.
    """
    torch.manual_seed(0)
    D, ref, bound = call_gemm(N=N, layout=layout, M=65, K=64, L=2)
    assert_close(D, ref, bound)


# ── validation that is not a point of the declared space ──────────────────────────────────────
@requires_sm90
@matrix_exempt("input validation of one flag/argument pair; nothing in the matrix varies it")
def test_dynamic_persistent_requires_a_semaphore():
    """The SM90 dynamic scheduler needs its GMEM ticket counter, and says so.

    There is no cluster-launch-control fallback on Hopper -- CLC is SM100 -- so without the counter
    there is nothing to hand out tiles, and the omission has to be an error rather than a silent
    fallback to the static scheduler.
    """
    A = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.empty(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"requires tile_count_semaphore"):
        gemm(A, B, D, None, None, 128, 128, 1, 1, is_dynamic_persistent=True)


@requires_sm90
@matrix_exempt("input validation of the dtype map; the unsupported dtypes are not in any pool")
def test_unsupported_dtype_is_named_not_a_keyerror():
    """An operand type outside the map is refused by name, listing what is supported.

    Indexing ``torch2cute_dtype_map`` directly would raise ``KeyError: torch.int8``, which tells
    the caller neither which operand nor what is allowed instead.
    """
    A = torch.zeros(1, 128, 128, device="cuda", dtype=torch.int8)
    B = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.empty(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"unsupported dtype for A: torch.int8"):
        gemm(A, B, D, None, None, 128, 128, 1, 1)


@requires_sm90
@matrix_exempt("input validation of tensor strides; the matrix has no misaligned values by design")
@pytest.mark.parametrize(
    "shape,expected",
    [
        ((1, 128, 201, 128), r"D violates the 16-byte alignment floor"),
        ((1, 128, 128, 129), r"A violates the 16-byte alignment floor"),
    ],
    ids=["N_not_multiple_of_8", "K_not_multiple_of_8"],
)
def test_misaligned_extents_are_refused_by_name(shape, expected):
    """The 16-byte floor is enforced at the front door, naming the operand and the multiple.

    Without this the violation surfaces from the TVM-FFI argument check as ``Invalid mD.strides[0]
    on argument #2 ... expected to be divisible by 8`` -- true, but it names an internal argument
    index rather than the extent the caller has to pad.
    """
    L, M, N, K = shape
    A = torch.randn(L, M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(L, N, K, device="cuda", dtype=torch.bfloat16)
    D = torch.empty(L, M, N, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=expected):
        gemm(A, B, D, None, None, 128, 128, 1, 1)


@requires_sm90
@matrix_exempt("a capability boundary, not a point of the supported space")
def test_stochastic_rounding_is_rejected_as_sm100_only():
    """``RoundingMode.RS`` is a Blackwell epilogue; on an SM90-only package it must say so."""
    A = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.empty(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(UnsupportedArchError, match="SM100"):
        gemm(A, B, D, None, None, 128, 128, 1, 1, rounding_mode=RoundingMode.RS)


@matrix_exempt("a capability boundary reached with CPU tensors; no kernel runs")
def test_gemm_rejects_a_non_sm90_device_at_the_front_door(monkeypatch):
    """A non-SM90 device fails at the entry with the boundary message, not three frames in.

    Upstream branched on the capability to pick between SM90/SM100/SM120 classes. Those classes are
    gone, so without the gate a B200 call would die on a ``NameError`` inside ``_compile_gemm``,
    with nothing actionable in the message. Driven with ``CPO_ARCH``
    and CPU tensors, so it runs without the silicon it is refusing.
    """
    import fold_cp_ops._internal.arch as arch

    monkeypatch.setenv("CPO_ARCH", "sm_100a")
    arch.get_device_capacity.cache_clear()
    try:
        A, B = torch.randn(1, 128, 128), torch.randn(1, 128, 128)
        D = torch.empty(1, 128, 128)
        with pytest.raises(UnsupportedArchError, match=r"requires SM90 \(H100/H200\)"):
            gemm(A, B, D, None, None, 128, 128, 1, 1)
    finally:
        arch.get_device_capacity.cache_clear()


@matrix_exempt("a property of the class hierarchy; there is nothing to parametrize")
def test_gemm_default_sm90_composes_the_mixin_before_the_base():
    """``GemmDefaultSm90``'s MRO puts the epilogue mixin ahead of ``GemmSm90``.

    The ordering is the entire content of the class: reversed, ``GemmSm90``'s no-op ``epi_*``
    defaults would win and the alpha/beta/bias terms would silently vanish -- every epilogue test
    above would then be comparing a plain ``A @ B`` against a reference that includes them, so this
    is really a statement about why those tests can fail at all.
    """
    from fold_cp_ops._internal.epi_default import GemmDefaultEpiMixin
    from fold_cp_ops.kernels.gemm_sm90 import GemmSm90

    mro = GemmDefaultSm90.__mro__
    assert mro.index(GemmDefaultEpiMixin) < mro.index(GemmSm90), (
        "the epilogue mixin must precede GemmSm90 in the MRO or its epi_* hooks never run"
    )


#: How each field of ``GemmSm90``'s two parameter packs reaches ``_compile_gemm``'s cache key.
#: A field maps to the key parameter that distinguishes it, to the key parameters it is DERIVED
#: from, or to ``None`` when it is pinned to a constant at this front door.
_PACK_FIELD_TO_KEY = {
    # ── construction phase
    "acc_dtype": None,  # pinned: `_compile_gemm` always builds with Float32
    "fp8_fast_accum": None,  # pinned: never passed, so always the constructor default
    "a_dtype": ("a_dtype",),
    "tile_shape_mn": ("tile_shape_mn",),
    "cluster_shape_mnk": ("cluster_shape_mnk",),
    "pingpong": ("pingpong",),
    "is_persistent": ("persistent",),
    # ── call phase
    "b_dtype": ("b_dtype",),
    "d_dtype": ("d_dtype",),
    "c_dtype": ("c_dtype",),
    "a_layout": ("a_major",),
    "b_layout": ("b_major",),
    "d_layout": ("d_major",),
    "c_layout": ("c_major",),
    "rounding_mode": ("rounding_mode",),
    "two_tensor_B": None,  # pinned: the plain GEMM has no second B operand
    # pinned: the plain GEMM has no second A operand either -- `gemm()` exposes no way to pass
    # one, so `a2_layout` is always None here. It is KEYED where it varies: the two-A x_gate
    # path has its own compile entry, which takes `a2_major`.
    "a2_layout": None,
    # derived, and therefore unable to differ between two calls that agree on their sources
    "cta_tile_k": ("a_dtype",),
    "epi_tile": ("tile_shape_mn", "d_dtype"),
    "ab_stage": ("tile_shape_mn", "a_dtype", "b_dtype", "d_dtype", "c_dtype"),
    "epi_stage": ("tile_shape_mn", "d_dtype"),
    "epi_c_stage": ("c_dtype",),
}


@matrix_exempt("relates two declarations to each other; there is no kernel launch to parametrize")
def test_the_jit_cache_key_covers_the_functor_s_entire_compile_time_surface():
    """Every field of ``Params``/``CallParams`` is keyed, derived from a keyed value, or pinned.

    **This is what makes the two parameter packs worth having.** Everything on ``self`` when
    ``__call__`` is traced is folded into the compiled kernel, so if two configurations differ in
    any pack field and the ``@jit_cache`` key cannot tell them apart, they share one ``.o`` -- and
    the kernel that runs is not the one that was selected. That failure is silent, survives process
    restarts through the persistent disk cache, and produces a wrong answer rather than an error.

    Before the packs existed the surface was not enumerable, so this could only be reasoned about.
    Now it is a set difference, and the map above is the place a new parameter must be classified:
    adding one without deciding whether it is keyed, derived or pinned fails here.

    The three classes are not equivalent, and the map records which is which:

    * **keyed** -- the value is a key argument, so two configurations differing in it cannot collide;
    * **derived** -- it is a pure function of keyed values, so it CANNOT differ when they agree;
    * **pinned** -- this front door never varies it. Safe only while that stays true, which is the
      other half of what this test checks: a pinned field must not be reachable from ``gemm()``.
    """
    import inspect

    from fold_cp_ops.kernels.gemm import _compile_gemm
    from fold_cp_ops.kernels.gemm_sm90 import GemmSm90

    key_params = set(inspect.signature(_compile_gemm.__wrapped__).parameters)
    surface = set(GemmSm90.Params.field_names()) | set(GemmSm90.CallParams.field_names())

    unclassified = surface - set(_PACK_FIELD_TO_KEY)
    assert not unclassified, (
        f"these compile-time parameters are not classified against the cache key: "
        f"{sorted(unclassified)}. Add each to _PACK_FIELD_TO_KEY as keyed, derived or pinned -- an "
        f"unkeyed, underived parameter means two configurations can share one compiled artifact."
    )
    stale = set(_PACK_FIELD_TO_KEY) - surface
    assert not stale, f"_PACK_FIELD_TO_KEY names fields that no longer exist: {sorted(stale)}"

    for field, sources in _PACK_FIELD_TO_KEY.items():
        if sources is None:
            continue
        missing = set(sources) - key_params
        assert not missing, (
            f"{field!r} is said to be keyed/derived via {sorted(missing)}, which _compile_gemm does "
            f"not take. Either the key lost an argument or the map is stale."
        )

    pinned = {f for f, s in _PACK_FIELD_TO_KEY.items() if s is None}
    reachable = set(inspect.signature(gemm).parameters)
    assert not (pinned & reachable), (
        f"{sorted(pinned & reachable)} are pinned in the cache key but SETTABLE through gemm(); a "
        f"caller varying one would silently reuse another configuration's compiled kernel"
    )


#: `main`'s SM90 pool, transcribed from `gemm_config._get_sm90_configs(epilogue=None)` rather than
#: recomputed: recomputing it from the same expression would pass for any expression.
_MAIN_SM90_TILES_COOP = {(256, 128), (256, 160), (256, 192), (256, 208), (128, 224), (128, 256)}
_MAIN_SM90_TILES_PINGPONG = {(128, 128), (128, 160), (128, 192), (128, 208), (192, 128)}
_MAIN_SM90_CLUSTERS = {(1, 2), (2, 1)}


@matrix_exempt("relates this tree's declared pool to main's; there is no kernel launch")
def test_the_tuning_pool_reproduces_mains_sm90_configs():
    """This tree's autotune pool is `main`'s, config for config.

    **A narrower pool is a perf regression no correctness test can see.** If the tuner here never
    tries the tile `main`'s tuner picked, a tuned build here is slower than a tuned build there and
    every other test still passes. A WIDER pool is the opposite problem: the comparison against
    `main` stops meaning anything, because the two are no longer choosing from the same set.

    One deliberate difference, asserted rather than assumed away: `main` also swept `swap_ab`,
    doubling its pool to 44. That is a feature of its `gemm_interface` layer, which this tree does
    not have -- `gemm()` has no such knob, so there is nothing here to sweep. The check below is on
    the 22 that remain.
    """
    from fold_cp_ops.kernels.gemm import GEMM_TUNING_SPACE

    expected = {
        (tile, pingpong, cluster)
        for tiles, pingpong in (
            (_MAIN_SM90_TILES_COOP, False),
            (_MAIN_SM90_TILES_PINGPONG, True),
        )
        for tile in tiles
        for cluster in _MAIN_SM90_CLUSTERS
    }
    got = {
        ((c["tile_M"], c["tile_N"]), c["pingpong"], (c["cluster_M"], c["cluster_N"]))
        for c in GEMM_TUNING_SPACE.configs()
    }
    assert got == expected, (
        f"the pool diverged from main's.\n  missing: {sorted(expected - got)}\n"
        f"  extra:   {sorted(got - expected)}"
    )
    assert len(got) == 22
    assert GEMM_TUNING_SPACE.excluded_count() == 122, (
        "the sparse pool must be an EXCLUSION from the full declared grid, not a shorter list of "
        "axis values -- otherwise a combination nobody considered looks the same as one considered "
        "and rejected. Cluster (1, 1) is the case that matters: it is IN the grid and excluded, "
        "with the measurement behind it written into `because=`"
    )
    domains = {a.name: set(a.values) for a in GEMM_TUNING_SPACE.axes}
    assert 1 in domains["cluster_M"] and 1 in domains["cluster_N"], (
        "cluster (1, 1) must be REACHABLE in the declared grid and removed by the exclusion, not "
        "absent from the axis values. Measured: it ties at the thin-K shape and loses by up to 11% "
        "elsewhere, so leaving it out is a decision -- and a decision has to be visible to be one"
    )


@matrix_exempt("every pool config must be CONSTRUCTIBLE; that is a property of the pool")
@pytest.mark.parametrize("persistent", [True, False])
def test_every_pool_config_builds_a_functor(persistent):
    """No candidate in the pool raises from `GemmSm90.__init__` for a request validity admits.

    A candidate that raises is not a slow config -- it is a CRASHED SWEEP, and under a collective a
    crashed sweep on one rank is a hang on its peers. `validity` exists to make that unreachable;
    this is what proves it does, on the one constraint that is request-dependent (pingpong needs a
    persistent grid).
    """
    import cutlass
    from cutlass import Float32

    from fold_cp_ops.kernels.gemm import GEMM_TUNING_SPACE, gemm_config_is_valid
    from fold_cp_ops.kernels.gemm_sm90 import GemmSm90

    admitted = [
        c
        for c in GEMM_TUNING_SPACE.configs()
        if gemm_config_is_valid(c, {"persistent": persistent})
    ]
    assert admitted, "validity emptied the pool, which leaves the kernel with nothing to run"
    for c in admitted:
        GemmSm90(
            Float32,
            cutlass.BFloat16,
            (c["tile_M"], c["tile_N"]),
            (c["cluster_M"], c["cluster_N"], 1),
            pingpong=c["pingpong"],
            is_persistent=persistent,
        )
    if not persistent:
        assert len(admitted) < len(GEMM_TUNING_SPACE.configs()), (
            "a non-persistent request must DROP the pingpong half; if it does not, either the "
            "validity rule stopped working or the pool lost its pingpong configs"
        )


@matrix_exempt("a property of the validity callback, which takes no shape")
def test_the_validity_rule_is_pure_and_rejects_only():
    """Validity may reject; it may not reorder, and it may not consult anything but its arguments.

    Purity is not a style preference here. The candidate list must be IDENTICAL on every rank -- a
    rule that consulted the device, the environment or a clock would give ranks different pools, and
    the consensus step would then compare timings for different kernels and pick a winner some ranks
    never measured. Under a collective that is a hang, not a wrong number.
    """
    import inspect

    from fold_cp_ops.kernels.gemm import GEMM_TUNING_SPACE, gemm_config_is_valid

    src = inspect.getsource(gemm_config_is_valid)
    for forbidden in ("os.environ", "torch.cuda", "time.", "random."):
        assert forbidden not in src, (
            f"gemm_config_is_valid consults {forbidden!r}; a non-pure validity rule gives ranks "
            f"different candidate pools, which is a hang under a collective"
        )
    request = {"persistent": True}
    once = [gemm_config_is_valid(c, request) for c in GEMM_TUNING_SPACE.configs()]
    twice = [gemm_config_is_valid(c, request) for c in GEMM_TUNING_SPACE.configs()]
    assert once == twice, "validity is not deterministic across calls"


@matrix_exempt("a property of the tuned wrapper's signature; no kernel launch")
def test_the_tuned_entry_refuses_a_pinned_tuned_knob():
    """Passing a tuned knob to the tuned entry is a `TypeError`, not a silent override.

    If a caller could pin `tile_M` while the tuner picked the cluster, the config the tuner
    MEASURED and the kernel that actually ran would differ -- and the cached winner would be
    recorded against a measurement that never happened.

    **Both the keyword and the POSITIONAL form are checked**, and the positional one is the reason
    this test grew: `tile_M` is `gemm`'s sixth argument, so that is how callers actually pass it. A
    check against keywords alone let it through, and it died several frames later as
    ``TypeError: got multiple values for argument 'tile_M'`` -- the symptom, not the mistake.
    """
    knobs = set(gemm.autotuner.space.configs[0].all_kwargs())
    assert knobs == {"tile_M", "tile_N", "pingpong", "cluster_M", "cluster_N"}, (
        "the axes must be named for the parameters they set, or a reader has to translate between "
        "the declared space and the signature"
    )
    a = torch.zeros(1, 8, 8, device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"tuned knobs and cannot also be passed"):
        gemm(a, a, a, None, None, pingpong=True, do_autotune=True)
    with pytest.raises(ValueError, match=r"tuned knobs and cannot also be passed"):
        gemm(a, a, a, None, None, 128, 128, 1, 1, do_autotune=True)


@matrix_exempt("scores the declared pools; there is no kernel launch to parametrize")
def test_gemm_matrix_is_diverse():
    """Every facet of every axis is straddled by its pool, above the entropy floor."""
    from fold_cp_ops.testing.kernel_matrix import MIN_FACET_ENTROPY

    weak = [
        s
        for axis in GEMM.axes
        for s in axis.diversity()
        if s.waived is None and s.entropy < MIN_FACET_ENTROPY
    ]
    assert not weak, "under-covered facets:\n  " + "\n  ".join(str(s) for s in weak)


@requires_sm90
@GEMM.parametrize("persistent")
def test_both_grid_schedules_produce_the_same_result(persistent):
    """One resident wave and one CTA per tile visit the same tiles; only the order differs.

    Each schedule is checked against the SAME exact reference, which is what makes the claim in the
    name an assertion rather than a comment. Before this, the test called ``call_gemm`` and threw
    its ``(D, ref, bound)`` away -- so it proved only that neither schedule crashed, and a
    persistent grid that skipped its last wave entirely would have passed. Found by the numeric
    coverage gate, which asks whether a sanctioned comparison RAN.
    """
    D, ref, bound = call_gemm(
        ab_dtype=torch.bfloat16,
        a_major="k",
        layout="contiguous",
        M=256,
        N=128,
        K=64,
        L=1,
        persistent=persistent,
    )
    assert_close(D, ref, bound)


@requires_sm90
@GEMM.parametrize(
    "M",
    only={"M": (2048, 4096, 8192, 12288)},
    because=(
        "the `workflow_N_token` facet is the whole subject, and M is the axis that carries it -- "
        "N and K are not swept here because the test's content is that they EQUAL M"
    ),
)
def test_the_back_half_einsum_is_square_at_every_workflow_token_count(M):
    """Op 7 at ``M == N == K == N_token``, for each declared token count.

    **The equality is the test, not the values.** ``M``, ``N`` and ``K`` are independent axes, so
    a pool can contain 2048 on all three while no cell ever pairs them -- the suite then reads as
    covering the back-half einsum without ever having measured one. An equality between axes cannot
    be expressed by pools; it has to be written as a cell. That generalises past this kernel: for
    any law of the form "these extents are the same extent", the CONSTRAINT is the thing to
    declare, because the product of the pools is not it.

    Concretely, op 7 is ``out[i,j,d] = sum_k a[i,k,d] * b[j,k,d]`` with ``i = j = k = N_token``, so
    it is O(N^3) and compute-bound at production sizes. A cell that let K drift from N would be an
    O(N^2) thin-K matmul instead -- a different kernel regime, comm-bound rather than compute-bound
    -- and CLAUDE.md forbids it for perf work precisely because every conclusion drawn from it
    would be about the wrong thing. Correctness is more forgiving, but a correctness suite that
    only ever ran the thin-K shape would leave the square one uncompiled.

    ``L = 1``, deliberately. In the workflow ``L = B * (D/cp)``, and batching multiplies the fp64
    reference by ``L`` while adding nothing to the shape law under test -- ``test_gemm_shapes``
    already carries a batched square cell at ``(1000, 1000, 1000, 16)``. At ``L = 1`` even the
    largest rung is affordable: operands are ``N**2`` elements each, so 302 MB in bf16 at 12288 and
    a 1.2 GB fp64 reference.

    Skipping is by runtime `torch.OutOfMemoryError` only, never a size estimate.

    Args:
        M: The token count; ``N`` and ``K`` are set equal to it.
    """
    torch.manual_seed(0)
    try:
        D, ref, bound = call_gemm(M=M, N=M, K=M, L=1, bias="row", alpha_mode="scalar")
        assert_close(D, ref, bound)
    except torch.OutOfMemoryError:
        pytest.skip(f"the square einsum at N_token={M} does not fit on this device")
    finally:
        torch.cuda.empty_cache()
