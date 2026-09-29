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

"""Tests for ``fold_cp_ops.kernels.gemm_hadamard`` -- the GEMM whose epilogue MULTIPLIES by C.

The kernel is the stock GEMM with one operator changed, so this module tests the three things that
change with it and nothing the stock GEMM's own tests already cover:

1. **The combine.** ``D = (alpha * (A @ B^T) + bias) ⊙ C``, with the gate applied AFTER every
   additive term. Getting the order wrong is not a rounding difference, it is a different function,
   and ``test_the_gate_multiplies_the_bias_too`` is what distinguishes them.
2. **The `const_expr` collapse.** With no bias and no C the emitted kernel must be the stock GEMM's,
   byte for byte. ``test_the_ungated_trace_is_the_stock_gemms_cubin`` asserts that on the compiled
   CODE SECTIONS rather than on a source reading, because the claim is about what the compiler chose
   to emit.
3. **The front door.** ``C`` is required, and every malformed tensor argument is refused with a
   sentence naming it -- all of it before anything is traced.

**The shape pools are the A2A-fused TriMul's own.** ``M`` carries the token-pair counts
``N_token**2 / cp`` at cp = 16 for ``N_token`` in 2048/4096/8192/12288, and ``N``/``K`` carry the
feature dims 128/256/384/512, because this kernel's production caller is reference ops 9 and 12:
``p_out = trin @ p_out_w^T + p_out_b`` immediately gated by ``gate3``. The largest of those is 27 GiB
of operands, which is why the shape test skips on a runtime `torch.OutOfMemoryError` rather than
gating on an estimate.
"""

import hashlib
import struct

import pytest
import torch

from fold_cp_ops.kernels.gemm_hadamard import gemm_hadamard
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    dtype_facets,
    front_door_raises,
    int_facets,
    matrix_exempt,
)
from fold_cp_ops.testing.numerics import assert_elementwise

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(
    _SM != 9, reason=f"gemm_hadamard() needs sm_90; this GPU is sm_{_SM}0"
)

_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)

#: fp32 unit roundoff and the per-dtype half-ulp, mirrored from `fold_cp_ops.testing.numerics`
#: rather than imported, because those are private there and this module needs a DIFFERENT error
#: model -- see :func:`_hadamard_bound`.
_U32 = 2.0**-24
_HALF_ULP = {torch.float16: 2.0**-11, torch.bfloat16: 2.0**-8, torch.float32: 2.0**-24}

# ── the declared test matrix for this kernel ───────────────────────────────────────────────────
# **Add a config HERE, not to one test.** tests/perf/test_benchmark_perf_gemm_hadamard.py imports
# this very object, so the configs that are timed and the ones that are correctness-tested cannot
# drift.
GEMM_HADAMARD = KernelMatrix(
    kernel="gemm_hadamard",
    axes=(
        Axis(
            name="ab_dtype",
            domain=(
                "16-bit (bfloat16/float16) or 8-bit (float8_e4m3fn/float8_e5m2) A and B -- the "
                "widths the SM90 WGMMA atom has a form for. float32 is in the pool because it is a "
                "KEY of torch2cute_dtype_map (D, C and the biases may be fp32) and so reaches the "
                "entry, where it must be refused as an OPERAND"
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
            name="c_major",
            domain=(
                "'n' or 'm' -- which axis of the per-element gate is contiguous. INDEPENDENT of "
                "D's major: C is its own TMA operand with its own descriptor, so a transposed gate "
                "needs no copy. Each is a distinct compiled kernel (c_major is in the cache key)"
            ),
            values=("n", "m"),
            facets={"n_major": lambda v: v == "n", "m_major": lambda v: v == "m"},
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
            name="bias",
            domain=(
                "'none' / 'row' ((l, n), broadcast down rows) / 'col' ((l, m), broadcast across "
                "columns) / 'both'. Each is a separate epilogue op and therefore a separate "
                "kernel. All of them add BEFORE the gate multiplies"
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
                "how the operands are laid out in memory, independent of their SHAPE. The 16-byte "
                "floor is on the row PITCH and the base ADDRESS, never on an extent -- and a "
                "contiguous tensor makes the pitch equal the extent, which hides the distinction. "
                "'padded_pitch' is what makes an odd or sub-floor extent legal; 'tight_off_floor' "
                "and 'unaligned_base' are the two ways to break it"
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
            name="persistent",
            domain=(
                "bool; True launches one resident wave that loops over work tiles, False one CTA "
                "per tile. The gate's TMA C-load rides the epilogue pipeline either way"
            ),
            values=(False, True),
            facets={"persistent": lambda v: v, "one_shot": lambda v: not v},
        ),
        Axis(
            name="M",
            domain=(
                "any positive int as an extent -- M is tiled and predicated, so 1, 7, 63, 301 and "
                "1001 all compute. It picks up the 16-byte floor ONLY when D or C is m-major, "
                "where M becomes a row pitch; that is a property of the layout, not of M"
            ),
            values=(
                1,
                7,
                63,
                128,
                301,
                1000,
                1001,
                4096,
                262144,
                1048576,
                4194304,
                9437184,
            ),
            facets=dict(
                int_facets(tile=128, big=1000, small=64),
                # M is the TOKEN-PAIR count of the A2A-fused TriMul back half: N_token**2 / cp.
                # These four are N_token in 2048/4096/8192/12288 at cp = 16, i.e. the shapes this
                # kernel actually runs at in production. Without the facet they are four large
                # numbers a future narrowing could drop without anything noticing.
                trimul_token_pairs=lambda v: v in (262144, 1048576, 4194304, 9437184),
                # The declared FRONT series, under the name the table uses. Added ALONGSIDE
                # the facet above rather than replacing it: that one is guarding these values
                # today, and a rename would drop a live guard to gain a name. Two names for
                # one set is the cheap side of that trade.
                workflow_M=lambda v: v in (262144, 1048576, 4194304, 9437184),
            ),
        ),
        Axis(
            name="N",
            domain=(
                "any positive int, as an EXTENT. The 16-byte floor lands on the row PITCH of D and "
                "C, not on N: torch.empty(l, m, 208)[:, :, :201] works and torch.empty(l, m, 201) "
                "does not, and the difference between them is the pitch"
            ),
            values=(1, 3, 8, 120, 128, 200, 201, 256, 384, 512, 1000),
            facets={
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                "off_grid": lambda v: v % 128 != 0,
                "tile_aligned": lambda v: v % 128 == 0,
                "odd": lambda v: v % 2 == 1,
                "sub_floor": lambda v: v % 8 != 0,
                # N is the OUTPUT feature dim D of TriMul ops 9/12 (p_out_w is (D, D)).
                "trimul_feature_dim": lambda v: v in (128, 256, 384, 512),
                # Same set under the workflow table's name, alongside rather than instead:
                # `trimul_feature_dim` is the live guard on these values today.
                "workflow_D": lambda v: v in (128, 256, 384, 512),
            },
        ),
        Axis(
            name="K",
            domain=(
                "any positive int, as an EXTENT, with the same pitch-vs-extent split as N. K is "
                "the feature dim contracted by ops 9/12, so the interesting values are the "
                "production feature dims and the ones no k-tile divides"
            ),
            values=(1, 3, 8, 128, 136, 192, 256, 384, 512, 1000),
            facets={
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                "sub_tile": lambda v: v < 64,
                "tile_aligned": lambda v: v % 64 == 0,
                "off_grid": lambda v: v % 128 != 0,
                "odd_feature_dim": lambda v: v in (136, 1000),
                "trimul_feature_dim": lambda v: v in (128, 256, 384, 512),
                # Same set under the workflow table's name, alongside rather than instead:
                # `trimul_feature_dim` is the live guard on these values today.
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
                # L = Dloc * B for the A2A-fused TriMul back einsum, where Dloc = D / cp; the
                # profiled grid runs D in {128, 256, 384} at cp up to 16 with B in {1, 2}, so the
                # reachable values are 8, 16, 32 and 48.
                "trimul_per_device": lambda v: v in (8, 16, 32, 48),
            },
        ),
        Axis(
            name="arg_fault",
            domain=(
                "a malformed TENSOR ARGUMENT, or 'none'. The values name a CALLER MISTAKE rather "
                "than a kernel configuration, and every non-'none' one must be refused at the "
                "front door with an API-level error. Both a dtype and an extent are in the pool "
                "because a caller can get either wrong on any argument, and which arguments the "
                "kernel feeds to a TMA versus reads with a broadcast load is not visible in its "
                "signature. 'C_missing' is here rather than on its own axis because an omitted "
                "gate is the same KIND of mistake: this kernel exists to apply C, so a None one "
                "must not read as an ungated result. The two 'operand_' values are the PARTNER "
                "faults: a property that is checked on one tensor and not on the tensor it must "
                "agree with. Both were latent before -- an unsupported A/B dtype reached the "
                "compile-key lookup as a bare KeyError, and a mismatched extent reached the "
                "TVM-FFI ABI check, which names an argument index and a traced symbol"
            ),
            values=(
                "none",
                "C_missing",
                "C_dtype",
                "C_extent",
                "rowvec_dtype",
                "rowvec_extent",
                "operand_dtype",
                "operand_extent",
            ),
            facets={
                "well_formed": lambda v: v == "none",
                "bad_dtype": lambda v: v.endswith("_dtype"),
                "bad_extent": lambda v: v.endswith("_extent"),
                "absent_operand": lambda v: v == "C_missing",
            },
        ),
    ),
    # ORDER MIRRORS `gemm_hadamard()`'s CHECK ORDER, and that is load-bearing rather than tidy:
    # `parametrize_unsupported` stops at the FIRST matching region, so a combo in several regions is
    # tested against whichever is listed first. List them out of order and the test asserts a
    # message the kernel does not produce. The sequence in the front door is: C presence,
    # check_tensor on D / C / rowvec / colvec, the tile shape, the semaphore, then
    # `describe_operands` (operand WIDTH, then layout PITCH, then BASE), then the broadcast pitch,
    # then the fp8 major. Move a check in the source and this tuple moves with it.
    # A contraction whose epilogue multiplies by a gate tensor; the gate is a product, not
    # a saturating activation, so nothing here is flattened by a tail.
    computes=("contraction", "fp8_operands"),
    unsupported=(
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                arg_fault == "C_missing"
                and ab_dtype is torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
            ),
            raises=ValueError,
            match=r"requires the per-element C \(gate\) tensor",
            reason=(
                "C is what this kernel is FOR: with no gate it is the plain `gemm`, and the two "
                "have different names precisely so an omitted argument cannot silently degrade one "
                "into the other. The trace with no C is byte-identical to the stock GEMM's -- "
                "test_the_ungated_trace_is_the_stock_gemms_cubin proves it -- which is exactly why "
                "reaching it through THIS entry has to be refused rather than allowed to compute"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                arg_fault == "C_dtype"
                and ab_dtype is torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
            ),
            raises=ValueError,
            match=r"unsupported dtype for C",
            reason=(
                "C is a TMA operand whose dtype is only a load width, so nothing downstream "
                "refuses it: before the front door checked it, the only thing that failed was the "
                "`torch2cute_dtype_map[C.dtype]` lookup building the compile key -- a bare KeyError "
                "naming a torch dtype and no argument"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                arg_fault == "C_extent"
                and ab_dtype is torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
            ),
            raises=ValueError,
            match=r"C must be",
            reason=(
                "the gate is a FULL per-element tensor and is not broadcast, so a C narrower than "
                "D does not stretch -- the TMA descriptor is built against D's extents and the "
                "kernel reads past the end of C on the trailing tiles, silently, since a mapped "
                "page follows"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                arg_fault == "rowvec_dtype"
                and ab_dtype is torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
            ),
            raises=ValueError,
            match=r"unsupported dtype for rowvec_bias",
            reason=(
                "a broadcast vector's dtype is only a LOAD WIDTH, so nothing downstream refuses it "
                "-- the same bare-KeyError failure the C dtype had, on a different argument"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                arg_fault == "rowvec_extent"
                and ab_dtype is torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
            ),
            raises=ValueError,
            match=r"rowvec_bias must be",
            reason=(
                "the row vector is broadcast along M with stride 0, so one shorter than N biases "
                "the tail columns with whatever follows it in memory -- and here that wrong bias "
                "is then MULTIPLIED by the gate, which neither cancels it nor makes it larger, so "
                "nothing about the output looks out of range"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                arg_fault == "operand_dtype"
                and ab_dtype is torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
            ),
            raises=ValueError,
            match=r"unsupported dtype for B",
            reason=(
                "A and B were the only tensors this entry did NOT run a membership check on -- "
                "they were left to `describe_operands`, whose first act is "
                "`torch2cute_dtype_map[t.dtype]`, so an unsupported operand dtype surfaced as a "
                "bare `KeyError: torch.float64`. That names a torch dtype and no argument, and "
                "`KeyError` is not in API_LEVEL_ERRORS, so nothing about it was actionable. The "
                "WIDTH check (16- or 8-bit) still happens later in `describe_operands`, which is "
                "why this region is about membership and the float32 region below is about width"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, a_major, layout, arg_fault: (
                arg_fault == "operand_extent"
                and ab_dtype is torch.bfloat16
                and a_major == "k"
                and layout == "contiguous"
            ),
            raises=ValueError,
            match=r"must contract over the same extent",
            reason=(
                "every operand was validated ALONE and none was compared to the others, so a "
                "caller who got one extent wrong reached the TVM-FFI ABI check and was told "
                "`Mismatched mB.shape[1] on argument #1`, which names a traced symbol and an FFI "
                "slot rather than the argument they passed. The same guard covers D's extent, "
                "which was previously checked only as a SIDE EFFECT of an optional bias being "
                "present -- a pass contingent on an unrelated argument, which reads as coverage"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype: ab_dtype == torch.float32,
            raises=ValueError,
            match=r"unsupported operand dtype torch.float32 for A",
            reason=(
                "the SM90 WGMMA atom has 16-bit and 8-bit forms only; there is no fp32 MMA on "
                "Hopper. fp32 is nonetheless a key of torch2cute_dtype_map -- D, C and the "
                "broadcast biases legitimately are fp32 -- so it passes the dtype-membership check "
                "and would otherwise be refused from inside tracing, tens of milliseconds later"
            ),
        ),
        Unsupported(
            where=lambda layout: layout == "tight_off_floor",
            raises=ValueError,
            match=r"violates the 16-byte alignment floor",
            reason=(
                "a CONTIGUOUS tensor whose trailing extent is off the floor has a row pitch off "
                "the floor too, and a TMA descriptor's stride must be a multiple of 16 bytes. The "
                "extent itself is fine -- the SAME extent under 'padded_pitch' computes correctly, "
                "which is what test_extent_is_free_when_the_pitch_is_padded asserts. Declaring "
                "both sides is the only way the pitch-vs-extent distinction is stated rather than "
                "assumed"
            ),
        ),
        Unsupported(
            where=lambda layout: layout == "unaligned_base",
            raises=ValueError,
            match=r"does not start on a 16-byte boundary",
            reason=(
                "a TMA descriptor's global address must be 16-byte aligned, which a slice at a "
                "non-multiple offset breaks. Without the front-door check this surfaced as "
                "`Misaligned Tensor data on argument #2` from the FFI, which names neither the "
                "operand the caller passed nor the fix"
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
    ),
)


# ── operand construction ───────────────────────────────────────────────────────────────────────
def _pad_to_floor(extent, dtype):
    """Round an extent up to the row pitch a TMA descriptor requires for ``dtype``.

    Args:
        extent: The logical extent. Must be positive.
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

    **The layout is applied to the ALLOCATION, not to the finished view.** A post-processing pass
    would copy into a fresh contiguous buffer, silently converting an m-major operand into a
    k-major one -- which changes the *major*, i.e. the very thing the layout axis is about.

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
              giving the same extent with a conforming pitch.
            * ``"unaligned_base"`` -- conforming pitch, but the view starts one element into the
              allocation, so the base address is off a 16-byte boundary.

    Returns:
        A tensor of exactly ``shape``.
    """
    want = shape[-1]
    if layout == "padded_pitch":
        alloc, lo = shape[:-1] + (_pad_to_floor(want, dtype),), 0
    elif layout == "unaligned_base":
        alloc, lo = shape[:-1] + (_pad_to_floor(want + 1, dtype),), 1
    else:
        alloc, lo = shape, 0
    if dtype in _FP8:
        t = (torch.randn(*alloc, device=device) / 4).to(dtype)
    else:
        t = torch.randn(*alloc, device=device, dtype=dtype)
    return t[..., lo : lo + want]


def _zeros(shape, dtype, device="cuda", layout="contiguous"):
    """A zeroed output view under the same layout rules as :func:`_gen`.

    D carries the layout as well as A, B and C, and for the two illegal layouts it is often D that
    trips them -- N lives on D's trailing axis, so ``tight_off_floor`` with an off-floor N is a
    D-side violation.

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


def _hadamard_bound(A, B, C, ref, alpha, out_dtype):
    """Per-element error allowance for ``D = (alpha * A@B^T + bias) ⊙ C`` stored as ``out_dtype``.

    **Why `numerics.epilogue_error_bound` is the wrong model here and is not reused.** That one
    assumes every epilogue term after the accumulator is exact and that the only rounding left is
    the store, which holds for ``+ beta*C + bias``. It does not hold for a MULTIPLY: the
    accumulator's error is *amplified by* ``|C|`` before it ever reaches the store, so a gate of
    magnitude 8 makes the same kernel eight times less accurate in absolute terms. Reusing the
    additive bound would under-allow exactly where the gate is large -- i.e. produce spurious
    failures that look like a kernel bug.

    The model, term by term, with ``u = 2**-24`` the fp32 unit roundoff:

    * ``acc_err = |alpha| * gamma_K * (|A| @ |B|^T)`` -- the standard sequential-summation bound,
      ``gamma_K = K*u/(1-K*u)``, scaled by alpha because the epilogue multiplies before adding.
      Pessimistic for WGMMA's tree order, which is the safe direction.
    * ``+ u * |pre|`` -- the fp32 rounding of the alpha scale and the bias adds, which matters only
      where the accumulation error is near zero (small K, exact operands).
    * ``* |C|`` -- the amplification above, then ``+ u * |ref|`` for the fp32 multiply itself.
    * ``+ half_ulp * (|ref| + that)`` -- the store's rounding, against the post-epilogue value.

    Args:
        A: ``(l, m, k)`` operand, any dtype in the pool.
        B: ``(l, n, k)`` operand.
        C: The gate, broadcastable to ``ref``'s shape. Its MAGNITUDE is what the bound needs.
        ref: The exact reference for the whole epilogue -- alpha, both biases and the gate applied.
        alpha: The accumulator scale actually used. Its magnitude is what matters; a sign is ignored.
        out_dtype: The dtype D is stored as. Must be a key of ``_HALF_ULP``; an fp8 output would
            need a term for the store's saturation and is not produced by these tests.

    Returns:
        An fp64 tensor of per-element allowances, shaped like ``ref``. Strictly positive wherever
        any term is non-zero, so a ratio against it is finite.

    Raises:
        ValueError: If ``K * u >= 1``, where the gamma formula has no meaning (needs K > 2**24).
        KeyError: If ``out_dtype`` is outside ``_HALF_ULP``.
    """
    K = A.shape[-1]
    if K * _U32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    gamma = (K * _U32) / (1.0 - K * _U32)
    abs_terms = A.double().abs() @ B.double().abs().transpose(-1, -2)
    pre_err = abs(float(alpha)) * gamma * abs_terms
    # |pre| <= |alpha| * abs_terms + |bias|; bounding it by the alpha-scaled abs_terms plus the
    # reference's own magnitude divided by |C| would need a division, so use the looser and
    # allocation-free `abs_terms` -- this term is a factor 2**-24 and never dominates.
    pre_err = pre_err + _U32 * abs(float(alpha)) * abs_terms
    absC = C.double().abs()
    absref = ref.double().abs()
    out_err = absC * pre_err + _U32 * absref
    half_ulp = _HALF_ULP[out_dtype]
    return out_err + half_ulp * (absref + out_err)


def call_gemm_hadamard(
    ab_dtype=torch.bfloat16,
    a_major="k",
    c_major="n",
    alpha_mode="one",
    bias="none",
    layout="contiguous",
    persistent=True,
    M=256,
    N=256,
    K=256,
    L=2,
    tile_M=128,
    tile_N=128,
    cluster_N=1,
    arg_fault="none",
    device="cuda",
):
    """Build operands for one point of the matrix, call ``gemm_hadamard()``, return what to compare.

    One helper rather than one per test because the axes interact: ``layout`` changes how every
    operand is allocated, ``a_major``/``c_major`` change stride order, ``bias`` decides which
    broadcast vectors exist, and the reference has to follow each of those. Splitting that across
    tests is how the reference and the call drift apart.

    Args:
        ab_dtype: Element type of A and B. fp8 forces a bf16 output and gate, since an fp8 D would
            measure the store's saturation instead of the MMA.
        a_major: ``"k"`` for a contiguous A, ``"m"`` for a genuine transposed view.
        c_major: ``"n"`` for a contiguous gate, ``"m"`` for a transposed one. Independent of D.
        alpha_mode: ``"one"`` / ``"scalar"`` / ``"tensor"``, matching the axis.
        bias: ``"none"`` / ``"row"`` / ``"col"`` / ``"both"``. A column bias makes M its row pitch,
            so it requires ``M % 4 == 0``.
        layout: One of the ``layout`` axis's values, applied to A, B, C and D alike. Under
            ``"tight_off_floor"`` the caller is expected to pass an N or K off the floor -- the
            layout names the *intent*, and a conforming extent under it is simply contiguous.
        persistent: Forwarded to ``gemm_hadamard()``.
        M: Rows. Free, except under a column bias or an m-major C.
        N: Columns. The floor lands on D's and C's row PITCH; under ``"padded_pitch"`` N is free.
        K: Contraction extent, same pitch-vs-extent split as N.
        L: Batch count.
        tile_M: CTA tile M.
        tile_N: CTA tile N.
        cluster_N: Cluster extent along N. Power of two, and ``cluster_M * cluster_N <= 8``.
        arg_fault: Which single argument to MALFORM, or ``"none"``. Every non-``"none"`` value is
            expected to make the call raise; see the declared regions.
        device: CUDA device.

    Returns:
        ``(D, ref, bound)`` -- the kernel's output, an fp64 reference of the same shape, and the
        per-element allowance from :func:`_hadamard_bound`.

    Raises:
        ValueError: Propagated from ``gemm_hadamard()``'s front door, which is what the
            unsupported-region test relies on.
        torch.OutOfMemoryError: Propagated; the shape test converts it to a skip.
    """
    out_dtype = torch.bfloat16 if ab_dtype in _FP8 else ab_dtype
    A = (
        _gen((L, M, K), ab_dtype, device, layout)
        if a_major == "k"
        else _gen((L, K, M), ab_dtype, device, layout).transpose(-1, -2)
    )
    B = _gen((L, N, K), ab_dtype, device, layout)
    C = (
        _gen((L, M, N), out_dtype, device, layout)
        if c_major == "n"
        else _gen((L, N, M), out_dtype, device, layout).transpose(-1, -2)
    )
    D = _zeros((L, M, N), out_dtype, device, layout)
    alpha = {"one": 1.0, "scalar": 2.5}.get(alpha_mode) or torch.tensor(
        [2.5], device=device, dtype=torch.float32
    )
    rv = _gen((L, N), torch.float32, device) if bias in ("row", "both") else None
    cv = _gen((L, M), torch.float32, device) if bias in ("col", "both") else None
    if arg_fault == "C_missing":
        C = None
    elif arg_fault == "C_dtype":
        C = torch.zeros(L, M, N, device=device, dtype=torch.float64)
    elif arg_fault == "C_extent":
        C = _gen((L, M, N + 8), out_dtype, device)
    elif arg_fault == "rowvec_dtype":
        rv = torch.zeros(L, N, device=device, dtype=torch.float64)
    elif arg_fault == "rowvec_extent":
        rv = torch.zeros(L, N + 1, device=device, dtype=torch.float32)
    elif arg_fault == "operand_dtype":
        # B rather than A, so the message names the operand that was actually malformed and the
        # cell is distinguishable from A's in a failure report.
        B = torch.zeros(L, N, K, device=device, dtype=torch.float64)
    elif arg_fault == "operand_extent":
        # B's contraction extent widened: A and B now disagree on K, which nothing compared before.
        B = _gen((L, N, K + 8), ab_dtype, device)

    gemm_hadamard(
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
    )

    a_scale = alpha.item() if torch.is_tensor(alpha) else alpha
    pre = a_scale * (A.double() @ B.double().transpose(-1, -2))
    if rv is not None:
        pre = pre + rv.double().unsqueeze(-2)
    if cv is not None:
        pre = pre + cv.double().unsqueeze(-1)
    ref = pre * C.double()
    return D, ref, _hadamard_bound(A, B, C, ref, a_scale, out_dtype)


def assert_close(D, ref, bound, what="gemm_hadamard"):
    """Compare against the exact reference element by element, against a per-element bound.

    Args:
        D: The kernel's output.
        ref: The exact fp64 reference for the whole epilogue -- alpha, both biases and the gate.
        bound: Per-element allowance, shaped like ``ref``. From :func:`_hadamard_bound`; do NOT
            substitute a scalar, which discards the reason the bound is a tensor.
        what: Short label for the failure message.

    Returns:
        None.

    Raises:
        AssertionError: On a non-finite output or any element outside its own bound, reported with
            the violating count and the three worst coordinates, values, bounds and ratios.
    """
    assert_elementwise(D, ref, bound, what=what)


# ── the combine: every term, and the ORDER they are applied in ─────────────────────────────────
@requires_sm90
@GEMM_HADAMARD.parametrize("alpha_mode", "bias")
def test_alpha_and_bias_modes(alpha_mode, bias):
    """alpha's three modes crossed with all four bias shapes, every one a different kernel.

    The full 3 x 4 product, with no ``because=`` to write: ``alpha_mode`` selects between folding
    the multiply out, baking a constant in, and emitting a load, and each ``bias`` value is a
    different epilogue op list. None of them is a value of one input, so a mode that stopped being
    folded away would show up nowhere else.
    """
    torch.manual_seed(0)
    D, ref, bound = call_gemm_hadamard(alpha_mode=alpha_mode, bias=bias)
    assert_close(D, ref, bound)


@requires_sm90
@matrix_exempt("the subject is the ORDER of two epilogue terms; no shape or dtype varies")
def test_the_gate_multiplies_the_bias_too():
    """``(acc + bias) ⊙ C``, not ``acc ⊙ C + bias`` -- the two differ by ``bias * (C - 1)``.

    Both orders pass a loose relative comparison at random inputs, which is why this test pins the
    distinction with operands chosen to separate them: ``A = 0`` makes the accumulator exactly zero,
    so the output is ``bias ⊙ C`` under the correct order and ``bias`` under the wrong one. With
    ``C = 2`` those differ by a factor of two everywhere, and the assertion is exact rather than
    approximate because every value involved is representable.
    """
    L, M, N, K = 1, 128, 128, 64
    A = torch.zeros(L, M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.zeros(L, N, K, device="cuda", dtype=torch.bfloat16)
    C = torch.full((L, M, N), 2.0, device="cuda", dtype=torch.bfloat16)
    D = torch.zeros(L, M, N, device="cuda", dtype=torch.bfloat16)
    rv = torch.arange(N, device="cuda", dtype=torch.float32).expand(L, N).contiguous()
    gemm_hadamard(A, B, D, C, None, 128, 128, 1, 1, rowvec_bias=rv)
    want = (rv.unsqueeze(-2) * 2.0).to(torch.bfloat16).expand(L, M, N)
    assert torch.equal(D, want), (
        "the gate must be applied AFTER the bias: with a zero accumulator the output is bias*C, "
        f"not bias. Got {D[0, 0, :4].tolist()}, expected {want[0, 0, :4].tolist()}"
    )


@requires_sm90
@matrix_exempt(
    "the subject is the swap_ab SCHEDULE knob, whose claim is that it changes no result; both "
    "shapes it needs are spelled out here because the SQUARE one is what makes the two broadcast "
    "vectors interchangeable, which is the only case where the knob can be wrong"
)
@pytest.mark.parametrize("shape", [(1, 512, 512, 512), (1, 4096, 256, 128)], ids=["square", "tall"])
@pytest.mark.parametrize("vectors", ["none", "row", "col", "both"])
def test_swap_ab_changes_no_result(shape, vectors):
    """``swap_ab`` computes ``B @ A^T`` into a transposed view and must land the same numbers.

    It is a pure performance knob -- `main` sweeps it as one and its tuner picks it at every cell
    inspected -- so the whole safety argument for putting it in the tuning pool is that the result
    does not move. That is exactly the kind of claim that fails silently: exchanging the operands
    also exchanges which broadcast vector rides which axis, and a wrong mapping still produces a
    plausible tensor.

    With NEITHER or ONE vector the two paths are **bit-identical**, and that is asserted exactly.
    With BOTH there is a real difference and it is not a defect: the epilogue adds
    ``acc + rowvec + colvec`` unswapped and ``acc + colvec + rowvec`` swapped, so fp32 rounding
    differs wherever those cancel. Measured over 134 M elements at (M=262144, D=512), that is
    0.0017% of elements, and both results are equidistant from an fp64 reference -- max absolute
    error 7.4280e-02 for each, identical to five significant figures. So the both-vectors case is
    asserted with a tolerance and the others exactly, which is the honest split.

    The SQUARE shape is not decoration. At ``M != N`` the two vectors have different lengths, so a
    mis-map is refused by the front door's own `check_tensor` and the kernel never runs; only at
    ``M == N`` is the wrong mapping shape-legal, and therefore silent.
    """
    L, M, N, K = shape
    torch.manual_seed(0)
    A = torch.randn(L, M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(L, N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    C = torch.randn(L, M, N, device="cuda", dtype=torch.bfloat16)
    rv = (
        torch.randn(L, N, device="cuda", dtype=torch.float32)
        if vectors in ("row", "both")
        else None
    )
    cv = (
        torch.randn(L, M, device="cuda", dtype=torch.float32)
        if vectors in ("col", "both")
        else None
    )

    outs = []
    for swap in (False, True):
        D = torch.empty(L, M, N, device="cuda", dtype=torch.bfloat16)
        gemm_hadamard(
            A,
            B,
            D,
            C,
            None,
            tile_M=128,
            tile_N=128,
            cluster_M=1,
            cluster_N=1,
            swap_ab=swap,
            rowvec_bias=rv,
            colvec_bias=cv,
        )
        outs.append(D)

    if vectors == "both":
        # The only case with two epilogue terms to reorder, so hold it to ACCURACY rather than to
        # bits -- and to accuracy against fp64, not against each other. Comparing the two paths
        # directly cannot distinguish "rounded differently" from "mis-mapped", because both put the
        # disagreement on the same near-zero cancelling elements; comparing each to the reference
        # can, since a mis-map moves one of them and rounding moves neither.
        ref = (
            A.double() @ B.double().mT + rv.double().unsqueeze(-2) + cv.double().unsqueeze(-1)
        ) * C.double()
        e0 = (outs[0].double() - ref).abs().max().item()
        e1 = (outs[1].double() - ref).abs().max().item()
        assert e1 <= 2 * e0 + 1e-6, (
            f"swap_ab must not be LESS accurate: max|err| vs fp64 is {e1:.4e} swapped against "
            f"{e0:.4e} unswapped. Equal-to-within-noise means addition order; materially worse "
            f"means the broadcast vectors are riding the wrong axes."
        )
        assert e0 <= 2 * e1 + 1e-6, "the unswapped path became the worse one, which is also a bug"
    else:
        assert torch.equal(outs[0], outs[1]), (
            f"swap_ab must not change the result with vectors={vectors!r}: there is no second "
            f"epilogue term to reorder, so the two paths must agree BIT for bit. "
            f"{int((outs[0] != outs[1]).sum())} of {outs[0].numel()} elements differ."
        )


@requires_sm90
@matrix_exempt("a structural claim about the declared pool; no kernel runs and no shape varies")
def test_the_swap_ab_axis_widens_only_this_entrys_pool():
    """``gemm_hadamard`` sweeps 44 configs and ``gemm`` still sweeps 22.

    The axis is opt-in (`gemm_tuning_space(swap_ab=True)`) because `gemm()` has no `swap_ab`
    parameter -- handing ITS tuner the axis would make every candidate a `TypeError` on the first
    tuned call. Reproducing `main`'s pool exactly is the point of the pool: narrower is a perf
    regression no correctness test can see, wider makes the comparison against `main` meaningless.
    """
    from fold_cp_ops.kernels.gemm import GEMM_TUNING_SPACE, gemm_tuning_space
    from fold_cp_ops.kernels.gemm_hadamard import GEMM_HADAMARD_TUNING_SPACE

    plain = len(GEMM_TUNING_SPACE.configs())
    widened = len(GEMM_HADAMARD_TUNING_SPACE.configs())
    assert plain == 22, f"gemm's pool must stay at main's 22; got {plain}"
    assert widened == 2 * plain, f"the swap_ab axis must exactly double the pool; got {widened}"
    assert "swap_ab" not in [a.name for a in gemm_tuning_space().axes]
    assert sum(c.get("swap_ab") is True for c in GEMM_HADAMARD_TUNING_SPACE.configs()) == plain


@requires_sm90
@GEMM_HADAMARD.parametrize(
    "ab_dtype",
    "a_major",
    drop={"a_major": ["m"], "ab_dtype": [torch.float32]},
    because=(
        "A stays k-major because fp8 requires it -- (fp8, m-major) is a declared unsupported "
        "region, and test_major_modes covers the 16-bit m-major path. float32 is itself a declared "
        "region: legal as a D, C and bias type, not as an OPERAND type, and a supported-path test "
        "cannot assert a correct answer for a combo the matrix says must raise."
    ),
)
def test_dtypes(ab_dtype, a_major):
    """All four operand types run the fused gate, fp8 included, at a 16-byte-aligned N and K.

    fp8 is the case worth having: the gate is applied in fp32 after the accumulator, so an fp8
    kernel's extra error must come from the MMA and not from the multiply. The bound widens for the
    operands' own quantization, not for the epilogue.
    """
    torch.manual_seed(0)
    widen = 8.0 if ab_dtype in _FP8 else 1.0
    D, ref, bound = call_gemm_hadamard(ab_dtype=ab_dtype, a_major=a_major, bias="row")
    assert_close(D, ref, bound * widen)


@requires_sm90
@GEMM_HADAMARD.parametrize(
    "a_major",
    "c_major",
    "ab_dtype",
    only={"ab_dtype": [torch.bfloat16, torch.float16]},
    because=(
        "major modes are a 16-bit-only question: every non-k-major fp8 operand is a declared "
        "unsupported region, so sweeping fp8 here would parametrize cells the matrix refuses."
    ),
)
def test_major_modes(a_major, c_major, ab_dtype):
    """A and the gate may each be contiguous along either axis, independently and of D.

    ``c_major`` is the one that is specific to this kernel: the gate is its own TMA operand with
    its own descriptor, so a transposed ``gate3`` costs a different compiled kernel and no copy.
    The product is 2 x 2 x 2 -- small, and each cell is a distinct compile key.
    """
    torch.manual_seed(0)
    D, ref, bound = call_gemm_hadamard(
        ab_dtype=ab_dtype, a_major=a_major, c_major=c_major, bias="row", alpha_mode="scalar"
    )
    assert_close(D, ref, bound)


@requires_sm90
@GEMM_HADAMARD.parametrize("persistent")
def test_both_grid_schedules_agree(persistent):
    """One resident wave and one CTA per tile visit the same tiles; only the order differs.

    Worth its own cell here rather than folding into another test because the gate's C-load rides
    the epilogue pipeline, and the epilogue pipeline's depth is what the two schedules differ in.
    """
    torch.manual_seed(0)
    D, ref, bound = call_gemm_hadamard(persistent=persistent, bias="row", M=256, N=128, K=64, L=1)
    assert_close(D, ref, bound)


# ── shapes: the 16-byte floor is the only constraint ──────────────────────────────────────────
def _assert_close_chunked(D, A, B, C, rv, cv, alpha, out_dtype, rows=None):
    """Compare in M-slices, so the fp64 reference of a 9.4M-row output fits in memory.

    Purpose
        The TriMul token-pair counts reach ``M = 9437184``. An fp64 reference for that at
        ``N = 512`` is 36 GiB and the fp64 ``|A| @ |B|^T`` behind the bound is another 36 GiB, so
        the *reference* is what would not fit -- the kernel's own operands are 27 GiB and run fine.
        Slicing along M is exact: every output row depends on one row of A and on the whole of B,
        so a slice's reference is the same numbers the full one would produce.

    Args:
        D: The kernel's output, ``(l, m, n)``.
        A: ``(l, m, k)`` operand.
        B: ``(l, n, k)`` operand.
        C: The gate, ``(l, m, n)``.
        rv: The ``(l, n)`` row bias, or None.
        cv: The ``(l, m)`` column bias, or None.
        alpha: The accumulator scale actually used, as a float.
        out_dtype: D's dtype.
        rows: Rows per slice, or None to pick one from L, N and K so a slice's fp64 temporaries stay
            around 256 MiB. The batch count is in that division because a slice keeps ALL batches;
            leaving it out sizes the slice for L = 1 and OOMs at L = 8. A caller passing a huge
            value gets the OOM it asked for.

    Returns:
        None.

    Raises:
        AssertionError: From :func:`assert_close`, on the first slice with a violation.
    """
    N, K = D.shape[-1], A.shape[-1]
    rows = rows or max(1, 2**25 // (max(N, K) * D.shape[0]))
    for lo in range(0, D.shape[-2], rows):
        hi = min(lo + rows, D.shape[-2])
        a, c, d = A[:, lo:hi], C[:, lo:hi], D[:, lo:hi]
        pre = alpha * (a.double() @ B.double().transpose(-1, -2))
        if rv is not None:
            pre = pre + rv.double().unsqueeze(-2)
        if cv is not None:
            pre = pre + cv[:, lo:hi].double().unsqueeze(-1)
        ref = pre * c.double()
        assert_close(d, ref, _hadamard_bound(a, B, c, ref, alpha, out_dtype), what="gemm_hadamard")


@requires_sm90
@GEMM_HADAMARD.parametrize(
    "M",
    "N",
    "K",
    "L",
    cells=[
        (1, 8, 8, 1),  # the degenerate floor
        (7, 120, 8, 2),  # M far below tile_M, off-grid N, minimum K
        (63, 128, 192, 3),  # M just under a tile, odd batch
        (301, 200, 136, 7),  # off-grid on BOTH extents, and a K no 64-wide k-tile divides
        (1001, 1000, 1000, 1),  # odd M, off-grid N and K, all three above the switch
        (1000, 384, 384, 2),  # non-power-of-two everywhere
        (4096, 512, 512, 1),  # large enough for the persistent grid to loop
        (128, 256, 128, 48),  # the largest batch, at production feature dims
        # --- the A2A-fused TriMul's own shapes: M = N_token**2 / cp at cp = 16, N = K = D ---
        (262144, 128, 128, 1),  # N_token = 2048,  D = 128
        (1048576, 256, 256, 1),  # N_token = 4096,  D = 256
        (4194304, 384, 384, 1),  # N_token = 8192,  D = 384
        (9437184, 512, 512, 1),  # N_token = 12288, D = 512 -- 27 GiB of operands
        (262144, 512, 512, 8),  # L = D/cp at D = 128, cp = 16, at the largest feature dim
    ],
    because=(
        "shapes are a list, not a product: the pools cross to ~10k cells, each a full recompile "
        "and several a multi-GiB allocation. The first 8 span the axes independently -- M below, "
        "at and over a tile and odd; N and K off-grid, sub-floor and large; L single, even, odd "
        "and the largest -- inside the 16-byte floor, with (301, 200, 136) off-grid on BOTH "
        "extents at once. The last 5 are the production shapes the generic cells cannot reach: "
        "the four token-pair counts against the four feature dims, plus the batched form. The "
        "reference is what constrains these, not the kernel, which is why they compare in slices."
    ),
)
def test_shapes(M, N, K, L):
    """The declared shape freedom is real through the host entry, gate included.

    ``bias="row"`` and not ``"both"``: a column bias is shaped ``(l, m)``, which makes M its row
    pitch and imposes a 4-element alignment on an extent that is otherwise entirely free. That
    interaction belongs to the bias, not to the shape sweep -- and this test could not carry
    M = 7, 63, 301 or 1001 at all with a column bias.

    The largest cell allocates 27 GiB of operands. A `torch.OutOfMemoryError` becomes a SKIP, per
    the standing rule that a memory *estimate* must never gate a test: an estimate is wrong on the
    next box, while the allocation failing is the fact itself.
    """
    torch.manual_seed(0)
    out_dtype = torch.bfloat16
    try:
        A = _gen((L, M, K), out_dtype)
        B = _gen((L, N, K), out_dtype)
        C = _gen((L, M, N), out_dtype)
        D = _zeros((L, M, N), out_dtype)
        rv = _gen((L, N), torch.float32)
        gemm_hadamard(A, B, D, C, None, 128, 128, 1, 1, rowvec_bias=rv, alpha=2.5)
        _assert_close_chunked(D, A, B, C, rv, None, 2.5, out_dtype)
    except torch.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        pytest.skip(f"M={M} N={N} K={K} L={L} does not fit on this device: {e}")


@requires_sm90
@GEMM_HADAMARD.parametrize(
    "N",
    "K",
    "layout",
    only={"N": [1, 3, 201], "K": [1, 3, 8], "layout": ["padded_pitch"]},
    because=(
        "the subject is exactly the extents that are NOT multiples of the 16-byte floor, under the "
        "one layout that makes them legal. The conforming extents are covered by test_shapes; the "
        "other three layouts are either that test's case or a declared region. K = 8 is in the "
        "selection as the conforming control: same layout, a pitch that needs no padding, so a "
        "failure at K = 1 or 3 is attributable to the extent rather than to padded_pitch itself."
    ),
)
def test_extents_are_free_when_the_pitch_is_padded(N, K, layout):
    """Odd and sub-floor N and K compute correctly once every row pitch is padded to the floor.

    The supported half of the pitch-vs-extent distinction, and the reason the ``tight_off_floor``
    region is a statement about the *pitch* rather than about an extent. The same N contiguous is
    refused and padded is correct, and the difference between them is the pitch.

    This is the FIRST PRINCIPLE as an executable assertion for this kernel: the gate adds a fourth
    TMA operand, and a fourth operand is a fourth chance for a port to acquire a shape constraint
    the stock GEMM does not have. N and K of 1 and 3 are two and six bytes, well under the 16-byte
    figure that is often restated as a constraint on the extent itself.

    ``bias="none"``: a ``(l, n)`` row bias at N = 3 has a row pitch of 3, which the epilogue's
    4-element vectorized broadcast load refuses -- a real constraint, but one on the BIAS and not on
    N, and ``test_colvec_bias_imposes_an_alignment_on_M`` is where that belongs.
    """
    torch.manual_seed(0)
    D, ref, bound = call_gemm_hadamard(N=N, K=K, layout=layout, M=64, L=1, bias="none")
    assert_close(D, ref, bound)


@requires_sm90
@matrix_exempt("an interaction between one epilogue term and one extent; nothing else varies")
def test_colvec_bias_imposes_an_alignment_on_M():
    """Passing a column bias makes M the bias's row pitch, so M picks up a 4-element floor.

    Not a second shape constraint on the kernel: M is free at every cell of ``test_shapes``. It is
    a constraint on the ``(l, m)`` bias tensor, which the epilogue loads with a 4-element vectorized
    copy. Worth a test because the failure otherwise reads ``Invalid epilogue_args[...].strides[0]
    ... expected to be divisible by 4``, naming neither the argument nor the extent to pad.
    """
    torch.manual_seed(0)
    with pytest.raises(ValueError, match=r"colvec_bias has stride"):
        call_gemm_hadamard(M=301, N=256, K=256, L=2, bias="col")
    D, ref, bound = call_gemm_hadamard(M=304, N=256, K=256, L=2, bias="col")
    assert_close(D, ref, bound)


# ── the declared unsupported regions ──────────────────────────────────────────────────────────
@requires_sm90
@GEMM_HADAMARD.parametrize_unsupported("ab_dtype", "a_major", "layout", "arg_fault")
def test_unsupported_combos_raise(
    ab_dtype, a_major, layout, arg_fault, expected_error, expected_match
):
    """Every declared unsupported combination is refused by ``gemm_hadamard()`` itself.

    All of these raise before any kernel is compiled, so the sweep stays cheap despite its size,
    and the operands are deliberately tiny -- nothing here computes.

    ``layout`` is in the sweep because two regions read it, and ``parametrize_unsupported`` refuses
    to run without every region's axes: a sweep that silently skipped them would report coverage it
    does not have. The N passed under ``tight_off_floor`` is 201 -- an extent legal with a padded
    pitch and illegal contiguous, which is the entire distinction the region encodes.

    The five ``arg_fault`` regions are guarded to the otherwise-clean cell (bf16, k-major,
    contiguous). Regions match first-wins, so malforming an argument on a cell that some OTHER
    region owns would make the kernel raise about the argument while the test asserted the other
    message.

    The assertion is on the raise and not on an xfail: an xfail is satisfied by a kernel that
    quietly accepts the combination and returns a wrong answer, which is the failure this matrix
    exists to prevent, whereas ``pytest.raises`` fails that kernel with "DID NOT RAISE".
    """
    torch.manual_seed(0)
    N = 201 if layout == "tight_off_floor" else 64
    # The fault is applied ONLY on the cell the fault regions claim. Regions match first-wins, so
    # malforming an argument on a bad-dtype or bad-layout cell would make the kernel raise about the
    # argument while the test asserted the other region's message.
    faultable = ab_dtype is torch.bfloat16 and a_major == "k" and layout == "contiguous"
    with front_door_raises(expected_error, expected_match):
        call_gemm_hadamard(
            ab_dtype=ab_dtype,
            a_major=a_major,
            layout=layout,
            arg_fault=arg_fault if faultable else "none",
            M=64,
            N=N,
            K=64,
            L=1,
            tile_M=64,
            tile_N=64,
        )


@requires_sm90
@matrix_exempt("front-door argument validation; there is no shape or dtype to sweep")
def test_a_missing_tile_shape_and_a_missing_semaphore_are_refused():
    """The two remaining front-door rules, both inherited from ``gemm()`` and both raising.

    Neither is a kernel configuration, so neither belongs on an axis: a missing tile shape means
    the caller wanted the tuner and did not say so, and a dynamic-persistent launch without its
    atomic counter would silently skip work tiles -- a truncated output rather than an error.
    """
    a = torch.zeros(1, 64, 64, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros(1, 64, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"tile_M and tile_N are required"):
        gemm_hadamard(a, a, c, c, None)
    with pytest.raises(ValueError, match=r"requires tile_count_semaphore"):
        gemm_hadamard(a, a, c, c, None, 64, 64, 1, 1, is_dynamic_persistent=True)


# ── the const_expr collapse, asserted on the compiled code ────────────────────────────────────
def _code_sections(object_file):
    """Map ``.text`` / ``.nv.constant*`` section KIND -> (size, sha256) for one exported artifact.

    Purpose
        `export_to_c` writes a HOST object with the CUDA ELF (the cubin) embedded in ``.lrodata``.
        Two artifacts are never byte-equal as whole files even when their code is identical,
        because the mangled kernel symbol carries the producing Python module and class
        (``...gemm_hadamardGemmHadamardSm90...`` vs ``...gemmGemmDefaultSm90...``) and the differing
        NAME LENGTH shifts every string that follows. What has to match is the code, and the code
        sections carry no names.

    Semantics
        Scans for the single embedded ELF whose ``e_machine`` is 190 (EM_CUDA), walks its section
        headers, and hashes the bytes of every PROGBITS section named ``.text*`` or
        ``.nv.constant*``. The mangled kernel suffix is stripped from each section name so the two
        sides' entries line up by KIND.

    Args:
        object_file: Path to a ``.o`` written by ``export_to_c``. Must contain exactly one CUDA ELF;
            a fat artifact with several would silently report only the first.

    Returns:
        ``{kind: (size, sha256_hex)}``.

    Raises:
        RuntimeError: If no CUDA ELF is embedded, which means the export produced something other
            than a compiled kernel and the comparison would be vacuous.
    """
    blob = open(object_file, "rb").read()
    i = blob.find(b"\x7fELF", 1)
    while i != -1 and struct.unpack_from("<H", blob, i + 18)[0] != 190:
        i = blob.find(b"\x7fELF", i + 1)
    if i == -1:
        raise RuntimeError(f"no CUDA ELF embedded in {object_file}")
    (shoff,) = struct.unpack_from("<Q", blob, i + 40)
    # ELF64 header: e_shentsize at +58, e_shnum at +60, e_shstrndx at +62.
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, i + 58)
    hdrs = [struct.unpack_from("<IIQQQQ", blob, i + shoff + n * shentsize) for n in range(shnum)]
    stroff = i + hdrs[shstrndx][4]
    out = {}
    for name, typ, _flags, _addr, off, size in hdrs:
        end = blob.index(b"\0", stroff + name)
        nm = blob[stroff + name : end].decode()
        if typ == 1 and size and (nm.startswith(".text") or nm.startswith(".nv.constant")):
            body = blob[i + off : i + off + size]
            out[nm.split(".kernel_")[0]] = (size, hashlib.sha256(body).hexdigest())
    return out


@requires_sm90
@matrix_exempt("the subject is the EMITTED CODE for one configuration, not a shape or a dtype")
def test_the_ungated_trace_is_the_stock_gemms_cubin(tmp_path):
    """With no gate and no bias, ``GemmHadamardSm90`` emits the stock GEMM's cubin, byte for byte.

    This is the claim the whole module rests on: every term in the Hadamard epilogue is
    `const_expr`-gated on PRESENCE, so a configuration with none of them present must collapse to
    the parent's trace. Reading the source cannot settle it -- the question is which instructions
    the compiler chose to emit -- so the comparison is on the compiled ``.text`` and
    ``.nv.constant0`` sections.

    **The negative control is half the test.** With the gate present the two must DIFFER; without
    it, a comparison that trivially passes (a section that is empty, or a name-keyed lookup that
    matched nothing on both sides) would be indistinguishable from a real result.

    Reaches ``_compile_gemm_hadamard`` directly rather than ``gemm_hadamard()``, because the
    ungated configuration is exactly the one the front door refuses -- and refuses on purpose; see
    the ``C_missing`` region.
    """
    from cutlass import BFloat16

    from fold_cp_ops._internal.rounding import RoundingMode
    from fold_cp_ops.kernels.gemm import _compile_gemm
    from fold_cp_ops.kernels.gemm_hadamard import _compile_gemm_hadamard

    cap, tile, cluster = (9, 0), (128, 128), (1, 1, 1)
    c_major_of = lambda dt: "n" if dt is not None else None  # noqa: E731

    def dump(compiled, name):
        p = tmp_path / f"{name}.o"
        compiled.export_to_c(object_file_path=str(p), function_name="kernel_entry")
        return _code_sections(str(p))

    for c_dtype, expect_same in ((None, True), (BFloat16, False)):
        # The two compile entries share a leading prefix -- (a, b, d, c dtypes), (a, b, d, c
        # majors), tile, cluster, then (pingpong, persistent, is_dynamic_persistent) -- and diverge
        # only in which epilogue modes each declares. Naming the common part keeps the divergence
        # readable instead of burying it in two twenty-argument calls.
        head = (BFloat16, BFloat16, BFloat16, c_dtype, "k", "k", "n", c_major_of(c_dtype))
        geom = (tile, cluster, False, True, False)
        had = dump(
            # alpha_mode, rowvec, colvec, colvec_ndim, batch_idx_permute, capacity
            _compile_gemm_hadamard(*head, *geom, 0, None, None, 0, False, cap),
            f"had_{c_dtype}",
        )
        stock = dump(
            # rowvec, colvec, colvec_ndim, alpha_mode, beta_mode, add_to_output,
            # batch_idx_permute, capacity, rounding_mode, sr_seed_mode, run_j_tiles
            _compile_gemm(
                *head, *geom, None, None, 0, 0, 0, False, False, cap, RoundingMode.RN, 0, 0
            ),
            f"stock_{c_dtype}",
        )
        assert had and ".text" in had, f"no code sections extracted: {had}"
        if expect_same:
            assert had == stock, (
                "the ungated Hadamard trace must be the stock GEMM's cubin. Differing sections: "
                + ", ".join(k for k in set(had) | set(stock) if had.get(k) != stock.get(k))
            )
        else:
            assert had != stock, (
                "NEGATIVE CONTROL FAILED: with the gate present the two cubins are identical, so "
                "the comparison above is not measuring the epilogue at all"
            )


@matrix_exempt("scores the declared pools; there is no kernel launch to parametrize")
def test_matrix_is_diverse():
    """Every facet of every axis is straddled by its pool, above the entropy floor."""
    from fold_cp_ops.testing.kernel_matrix import MIN_FACET_ENTROPY

    weak = [
        s
        for axis in GEMM_HADAMARD.axes
        for s in axis.diversity()
        if s.waived is None and s.entropy < MIN_FACET_ENTROPY
    ]
    assert not weak, "under-covered facets:\n  " + "\n  ".join(str(s) for s in weak)


@matrix_exempt("a property of the tuned entry's declared space, not of any shape")
def test_the_tuned_entry_names_the_axes_it_sets():
    """The tuning axes are named for the parameters they set, and the gate refuses a pinned knob.

    Both halves matter. If the axis names drifted from the signature a reader would have to
    translate between the declared space and the arguments; and if a caller could pin ``tile_M``
    while the tuner picked the cluster, the config that was MEASURED and the kernel that RAN would
    differ, with the cached winner recorded against a measurement that never happened.

    ``swap_ab`` is in the set because this entry implements the swap and therefore opts into the
    axis; `gemm()` does not and its pool stays five-wide. The pinning half covers it too -- a knob
    the tuner sweeps must be refused when a caller also passes it, and `swap_ab` is now such a knob.
    """
    knobs = set(gemm_hadamard.autotuner.space.configs[0].all_kwargs())
    assert knobs == {"tile_M", "tile_N", "pingpong", "cluster_M", "cluster_N", "swap_ab"}, knobs
    a = torch.zeros(1, 8, 8, device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"tuned knobs and cannot also be passed"):
        gemm_hadamard(a, a, a, a, None, swap_ab=True, do_autotune=True)
    with pytest.raises(ValueError, match=r"tuned knobs and cannot also be passed"):
        gemm_hadamard(a, a, a, a, None, pingpong=True, do_autotune=True)
    with pytest.raises(ValueError, match=r"tuned knobs and cannot also be passed"):
        gemm_hadamard(a, a, a, a, None, 128, 128, 1, 1, do_autotune=True)
