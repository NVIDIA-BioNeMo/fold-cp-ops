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

"""Tests for ``fold_cp_ops.kernels.dual_gated_gemm`` -- the SM90 dual-gated GEMM.

**The strongest gates here need no reference at all.** The gate is nonlinear, so a comparison
against torch must carry a derived tolerance (`gated_error_bound`), and a tolerance can only say
"close enough". But several of the kernel's degrees of freedom are pure transport -- they change
which register holds a value, or which lane stores it, and must not change the arithmetic at all.
For those the gate is `torch.equal`:

* the two weight layouts (`chunk_g` 1 vs 16) pair different registers to compute the same product;
* the transposed store writes the same values through a different descriptor;
* persistent, non-persistent and ping-pong schedules visit the same tiles in a different order but
  accumulate each one identically.

A bitwise disagreement on any of those is a real defect that a tolerance test would absorb, which
is why they are separated from the reference comparison rather than folded into it.

**On the reference tolerance.** At bf16 the dominant term is NOT the accumulation -- it is the
hardware sigmoid's ``2**-12`` absolute error, which `gated_error_bound` carries through the gate's
slope. That is why a gated kernel legitimately shows more error than the plain GEMM underneath it.
"""

import time

import cutlass
import pytest
import torch

from fold_cp_ops.kernels.gemm_sm90 import GemmSm90Params
from fold_cp_ops.kernels.dual_gated_gemm import (
    DualGatedGemmParams,
    DualGatedGemmSm90,
    _interleave_dual_bias_torch,
    _interleave_dual_weights_torch,
    append_gate3_weight,
    build_dual_operands,
    dual_gated_gemm,
    dual_gated_gemm_ref,
    interleave_dual_weights,
)
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    dtype_facets,
    front_door_raises,
    matrix_exempt,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt
from fold_cp_ops.testing.numerics import (
    assert_bitwise,
    assert_elementwise,
    gated_error_bound,
    sigmoid_error_bound,
    tolerance_bound,
)

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(
    _SM != 9, reason=f"dual_gated_gemm needs sm_90; this GPU is sm_{_SM}0"
)

DUAL_GATED = KernelMatrix(
    kernel="dual_gated_gemm",
    axes=(
        Axis(
            name="ab_dtype",
            domain=(
                "bfloat16 or float16 for the activation, both weights and the gated output -- the "
                "16-bit widths the SM90 WGMMA atom has a form for AND the stmatrix store leg has a "
                "form for. float32 is in the pool because it is a KEY of torch2cute_dtype_map (the "
                "biases legitimately are fp32) and so reaches the entry, where it must be refused "
                "as an operand. fp8 is absent from the pool, not merely untested: the gated store "
                "is stmatrix-based and has no 8-bit form"
            ),
            values=(torch.bfloat16, torch.float16, torch.float32),
            facets=dtype_facets((torch.bfloat16, torch.float16, torch.float32)),
        ),
        Axis(
            name="chunk_g",
            domain=(
                "1 (element interleave: the host interleaves the two weights, the epilogue pairs "
                "adjacent registers) or a multiple of 16 (block interleave: the weights are loaded "
                "directly by a two-tensor TMA, the epilogue pairs across a register half-split). "
                "8 is in the pool because it is the tempting value -- half the stmatrix atom width "
                "-- and must be refused rather than silently mis-paired"
            ),
            values=(1, 8, 16),
            facets={
                "element_interleave": lambda v: v == 1,
                "block_interleave": lambda v: v > 1 and v % 16 == 0,
                "sub_atom": lambda v: 1 < v < 16,
            },
        ),
        Axis(
            name="bias",
            domain=(
                "'none', 'both' (a bias on each projection), or 'gate_only'. The last is in the "
                "pool because biasing one projection is a plausible typo that the kernel cannot "
                "detect downstream -- the pre-activation would simply be half-biased"
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
                "be folded into the biases -- the gate is nonlinear -- so it is a distinct epilogue "
                "term, not a scaling of an existing one"
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
            name="persistent",
            domain=(
                "bool; True launches one resident wave that loops over work tiles, False one CTA "
                "per tile. Required by pingpong"
            ),
            values=(False, True),
            facets={"persistent": lambda v: v, "one_shot": lambda v: not v},
        ),
        Axis(
            name="tile_n",
            domain=(
                "CTA tile over the 2N PRE-activation, so the output tile is half of it. A multiple "
                "of 16 and <= 256, or a multiple of 32 and <= 512 (GemmSm90's own floor), AND a "
                "multiple of 2*chunk_g in the block-interleave layout. It does NOT have to divide "
                "the feature extent: a partial last tile is predicated, which is what keeps the "
                "extent free"
            ),
            values=(32, 64, 128, 256),
            facets={
                "narrow": lambda v: v <= 64,
                "wide": lambda v: v >= 128,
                "widest": lambda v: v == 256,
            },
        ),
        Axis(
            name="cluster_n",
            domain=(
                "threadblock-cluster extent along N; a power of two with cluster_M*cluster_N <= 8. "
                "> 1 multicasts B across the cluster, which the two-tensor gated load cannot do"
            ),
            values=(1, 2),
            facets={"clustered": lambda v: v > 1, "single_cta": lambda v: v == 1},
        ),
        Axis(
            name="gate3",
            domain=(
                "bool; whether the fused TriMul output gate is built. Its weight's rows are "
                "APPENDED to the dual weight, so an output-gate tile is an ORDINARY work tile of "
                "the same operand -- but the region boundary and the second store are compile-time "
                "facts, so it is a different kernel and not a runtime option"
            ),
            values=(False, True),
            facets={"gated_output": lambda v: v, "dual_only": lambda v: not v},
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
            name="M",
            domain=(
                "token extent, entirely free -- no tile multiple, no power of two. The pool spans "
                "the partial-tile case (below one tile_M), the off-grid cases, and the MULTI-WAVE "
                "regime the persistent scheduler only reaches past ~132 CTAs' worth of tiles, "
                "which is where a per-CTA scratch reused across tiles is exercised at all"
            ),
            values=(64, 129, 1000, 4096, 32768, 262144, 1048576, 4194304),
            facets={
                "partial_tile": lambda v: v < 128,
                "tile_multiple": lambda v: v % 128 == 0,
                "off_grid": lambda v: v % 128 != 0,
                "multi_wave": lambda v: v >= 32768,
                # The A2A-fused TriMul workflow's own shard sizes: its front consumes
                # N_token**2 / cp rows, so at cp=16 the N_token ladder 2048 / 4096 / 8192 is
                # M = 262144 / 1048576 / 4194304. Mirrors `test_layernorm_dual_gated_gemm.py`'s
                # facet of the same name ON PURPOSE: the two perf gates time the SAME cells so
                # their pins subtract to the fusion's cost, and that subtraction is only
                # meaningful while both draw their range from the same ladder.
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
                "width cannot satisfy the diversity check"
            ),
            values=(64, 128, 137, 256, 384, 512),
            facets={
                "narrow": lambda v: v <= 64,
                "off_grid": lambda v: v % 128 != 0,
                "workflow_D": lambda v: v in (128, 256, 384, 512),
            },
        ),
        Axis(
            name="K",
            domain=(
                "contraction extent, and the ONLY extent carrying a constraint: the 16-byte "
                "alignment floor. The pool is the workflow's feature widths plus a non-power-of-two "
                "that still meets the floor, because K is where a tile-multiple assumption would "
                "hide -- plus 136, which meets the floor and NOTHING else: no allowed k-tile "
                "divides it, so it is the only value that reaches the PADDED contraction at all. "
                "The other five are every one a multiple of 64. That is not a hypothetical gap: "
                "the same pool shape on the FUSED sibling hid two crashes from 2857 tests, and a "
                "domain string claiming a non-power-of-two covered it read as satisfied while the "
                "property went untested -- 192 and 384 are non-powers of two AND multiples of 64"
            ),
            values=(128, 136, 192, 256, 384, 512),
            facets={
                "workflow_D": lambda v: v in (128, 256, 384, 512),
                "non_power_of_two": lambda v: v & (v - 1) != 0,
                "widest": lambda v: v >= 512,
                # The property the other five cannot express: K % blk_k != 0 for every allowed
                # blk_k, so the last k-tile is partial and the TMA must zero-fill it.
                "no_k_tile_divides": lambda v: v % 64 != 0 and v % 32 != 0 and v % 16 != 0,
            },
        ),
    ),
    # Order matters and mirrors the front door's check order: regions are matched first-wins, so a
    # combo violating two rules reports the one listed first here.
    # Two contractions combined through a sigmoid gate.
    computes=("contraction", "saturating_activation"),
    unsupported=(
        Unsupported(
            where=lambda ab_dtype: ab_dtype == torch.float32,
            raises=ValueError,
            match=r"must be 16-bit",
            reason=(
                "the gated post-activation is stored through stmatrix, which has a 16-bit form "
                "only, and the SM90 WGMMA atom has no fp32 form either. fp32 nonetheless passes "
                "the dtype-membership check -- it is a legitimate BIAS dtype -- so without this "
                "guard it would be refused several frames into tracing rather than at the door"
            ),
        ),
        Unsupported(
            where=lambda chunk_g: chunk_g != 1 and chunk_g % 16 != 0,
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
            where=lambda bias: bias == "gate_only",
            raises=ValueError,
            match=r"both be given or both omitted",
            reason=(
                "a bias on one projection only biases half the pre-activation. Nothing downstream "
                "can detect it: the kernel would add the gate bias, leave the up projection "
                "unbiased, and gate the two together into a result that looks entirely reasonable"
            ),
        ),
        Unsupported(
            where=lambda chunk_g, cluster_n: chunk_g > 1 and chunk_g % 16 == 0 and cluster_n > 1,
            raises=ValueError,
            match=r"requires cluster_N=1",
            reason=(
                "the block-interleave layout loads Wg and Wp through a two-tensor TMA that has no "
                "multicast form, so a cluster along N would have every CTA load the full weight "
                "instead of its share -- correct but pointless, and the descriptor build fails "
                "first. Refused at the door so the caller sees which knob to change"
            ),
        ),
        Unsupported(
            where=lambda ab_dtype, chunk_g, bias, cluster_n, gate3, arg_fault: (
                ab_dtype == torch.bfloat16
                and chunk_g == 1
                and bias == "both"
                and cluster_n == 1
                and not gate3
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
            where=lambda ab_dtype, chunk_g, bias, cluster_n, gate3, arg_fault: (
                ab_dtype == torch.bfloat16
                and chunk_g == 1
                and bias == "both"
                and cluster_n == 1
                and not gate3
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
            where=lambda ab_dtype, chunk_g, bias, cluster_n, gate3, arg_fault: (
                ab_dtype == torch.bfloat16
                and chunk_g == 1
                and bias == "both"
                and cluster_n == 1
                and not gate3
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
            where=lambda ab_dtype, chunk_g, bias, cluster_n, gate3, arg_fault: (
                ab_dtype == torch.bfloat16
                and chunk_g == 1
                and bias == "both"
                and cluster_n == 1
                and not gate3
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

#: The TriMul input projection's shape family: the activation is (tokens, hidden) and each
#: projection is (features, hidden). Small enough to keep the suite quick, off-grid enough that the
#: partial-tile paths run.
_M, _N, _K = 256, 128, 128


def _pitch_pad(rows, cols, dtype, fill=0.0):
    """Allocate a ``(rows, cols)`` view whose row pitch meets the 16-byte TMA floor.

    Purpose
        The floor is on the PITCH, not the extent (ledger 3.9): a contiguous ``(M, 137)`` bf16
        tensor has a 274-element row stride, which is not a multiple of 8, so its TMA descriptor
        cannot be built. Padding the pitch leaves the extent completely free, which is what lets
        these tests use genuinely off-grid N.

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


def _build(M=_M, N=_N, K=_K, dtype=torch.bfloat16, bias="none", mask=False, N3=0, seed=0):
    """Build one cell's operands.

    Args:
        M: Token extent. Unconstrained -- off-grid values are the interesting ones.
        N: Output feature extent, i.e. HALF the pre-activation width. Unconstrained.
        K: Contraction extent. Must meet the 16-byte floor on the operands' row pitch.
        dtype: Operand and output element type; one of the pool's 16-bit values.
        bias: ``"none"``, ``"both"``, or ``"gate_only"`` (which the kernel must refuse).
        mask: Whether to build a per-row post-gate mask.
        N3: Output-gate width, or 0 for no output gate. When non-zero the return grows by
            ``(W3, b3)`` -- appended rather than inserted, so every existing unpack keeps working.
        seed: RNG seed, so a failure is reproducible.

    Returns:
        ``(A, Wg, Wp, bg, bp, mk)``, or ``(A, Wg, Wp, bg, bp, mk, W3, b3)`` when `N3` is non-zero.
        The bias and mask entries are None when not requested. Weights
        are scaled by 0.1 so the pre-activations sit in the sigmoid's responsive range rather than
        saturating, where every gate value would be 0 or 1 and the test would pass vacuously.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(M, K, device="cuda", dtype=dtype, generator=g)
    Wg = torch.randn(N, K, device="cuda", dtype=dtype, generator=g) * 0.1
    Wp = torch.randn(N, K, device="cuda", dtype=dtype, generator=g) * 0.1
    bg = bp = None
    if bias in ("both", "gate_only"):
        bg = torch.randn(N, device="cuda", dtype=torch.float32, generator=g)
    if bias == "both":
        bp = torch.randn(N, device="cuda", dtype=torch.float32, generator=g)
    mk = torch.rand(M, device="cuda", dtype=dtype, generator=g) if mask else None
    if not N3:
        return A, Wg, Wp, bg, bp, mk
    W3 = torch.randn(N3, K, device="cuda", dtype=dtype, generator=g) * 0.1
    b3 = torch.randn(N3, device="cuda", dtype=torch.float32, generator=g)
    return A, Wg, Wp, bg, bp, mk, W3, b3


def _out_for(M, N, dtype, transpose_out):
    """Allocate the gated output in the requested majorness, pitch-padded either way.

    The logical shape is ``(M, N)`` in both cases; m-major is a transposed VIEW, not a different
    shape. Passing a raw ``(N, M)`` tensor would be a shape mismatch, not a transposed store.
    """
    return _pitch_pad(N, M, dtype).T if transpose_out else _pitch_pad(M, N, dtype)


def _run(
    A, Wg, Wp, out, *, chunk_g=1, bg=None, bp=None, mask=None, tile_N=None, W3=None, b3=None, **kw
):
    """Call the kernel with this file's default tile geometry.

    Args:
        A, Wg, Wp, out: The operands; `out` is written in place.
        chunk_g: Weight layout.
        bg, bp, mask: Optional epilogue terms.
        tile_N: CTA tile over the 2N PRE-activation. Defaults to 128, or to ``2*chunk_g`` when that
            is smaller, since the block-interleave layout requires tile_N to be a multiple of it.
        W3, b3: The optional output gate. When `W3` is given a destination is allocated here and
            returned alongside `out`.
        **kw: Forwarded (persistent, pingpong, cluster_M, cluster_N, ...).

    Returns:
        `out`, or ``(out, gate3)`` when `W3` is given.
    """
    if tile_N is None:
        tile_N = 128
    out3 = None if W3 is None else _pitch_pad(A.shape[0], W3.shape[0], out.dtype)
    dual_gated_gemm(
        A,
        Wg,
        Wp,
        out,
        tile_M=128,
        tile_N=tile_N,
        chunk_g=chunk_g,
        bg=bg,
        bp=bp,
        mask=mask,
        W3=W3,
        b3=b3,
        PostAct3=out3,
        **kw,
    )
    torch.cuda.synchronize()
    return out if out3 is None else (out, out3)


# ------------------------------------------------------------------ reference comparison


@requires_sm90
@DUAL_GATED.parametrize(
    "ab_dtype",
    "chunk_g",
    "bias",
    "mask",
    drop={"ab_dtype": (torch.float32,), "chunk_g": (8,), "bias": ("gate_only",)},
    because=(
        "the dropped values are exactly the declared unsupported regions, which "
        "test_dual_gated_gemm_unsupported_combos_raise sweeps instead -- they have no correct "
        "output to compare against"
    ),
)
def test_dual_gated_gemm_matches_the_reference(ab_dtype, chunk_g, bias, mask):
    """The gated output matches an fp32 reference within the derived per-element bound."""
    A, Wg, Wp, bg, bp, mk = _build(dtype=ab_dtype, bias=bias, mask=mask)
    out = _out_for(_M, _N, ab_dtype, False)
    _run(A, Wg, Wp, out, chunk_g=chunk_g, bg=bg, bp=bp, mask=mk)
    ref = dual_gated_gemm_ref(A, Wg, Wp, bg, bp, mk).double()
    assert_elementwise(out, ref, gated_error_bound(A, Wg, Wp, ref, ab_dtype))


@requires_sm90
@DUAL_GATED.parametrize(
    "chunk_g",
    "tile_n",
    only={"chunk_g": (1, 16)},
    because="8 is the declared unsupported value and has no correct output",
)
@pytest.mark.parametrize(
    "M,N,K",
    [(250, 137, 128), (301, 97, 192), (64, 16, 128), (513, 256, 256)],
    ids=["offgrid", "odd_all", "tiny_N", "large"],
)
def test_dual_gated_gemm_accepts_off_grid_and_odd_extents(chunk_g, tile_n, M, N, K):
    """Every extent is free; only the 16-byte PITCH floor constrains anything.

    N=137 and N=97 are neither powers of two nor tile multiples nor multiples of 8; M=250 and 301
    are off-grid; K=192 is not a power of two. The outputs are pitch-padded, which is the documented
    way to keep the extent arbitrary -- and the reason this passes where a contiguous allocation
    would be refused.

    Swept across the whole tile pool rather than one frozen tile, because a partial last tile is
    the case a single well-chosen tile would never exercise: the tile does NOT have to divide the
    feature extent, and this is where that claim is checked instead of assumed.
    """
    A, Wg, Wp, bg, bp, mk = _build(M, N, K, bias="both", mask=True)
    out = _out_for(M, N, torch.bfloat16, False)
    _run(A, Wg, Wp, out, chunk_g=chunk_g, bg=bg, bp=bp, mask=mk, tile_N=tile_n)
    ref = dual_gated_gemm_ref(A, Wg, Wp, bg, bp, mk).double()
    assert_elementwise(out, ref, gated_error_bound(A, Wg, Wp, ref, torch.bfloat16))


# ------------------------------------------------------------------ bitwise transport gates


@requires_sm90
@DUAL_GATED.parametrize(
    "bias",
    "mask",
    drop={"bias": ("gate_only",)},
    because="the half-biased cell has no correct output to be identical to",
)
def test_the_two_weight_layouts_are_bitwise_identical(bias, mask):
    """`chunk_g` 1 and 16 pair different registers to compute the same product -- bit for bit.

    The single strongest gate in this file. The two layouts share the mainloop and the accumulation
    order and differ only in which two accumulator registers the gate combines; if the pairing is
    right in both, the results are identical, and if it is wrong in either they are not. No
    reference and no tolerance are involved, so a mis-pairing cannot hide inside a bound.
    """
    A, Wg, Wp, bg, bp, mk = _build(bias=bias, mask=mask)
    o1 = _run(A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, False), chunk_g=1, bg=bg, bp=bp, mask=mk)
    o16 = _run(
        A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, False), chunk_g=16, bg=bg, bp=bp, mask=mk
    )
    assert_bitwise(o16, o1, what="chunk_g=16 vs chunk_g=1")


@requires_sm90
@DUAL_GATED.parametrize(
    "chunk_g",
    only={"chunk_g": (1, 16)},
    because="8 is the declared unsupported value",
)
def test_the_transposed_store_writes_the_same_values(chunk_g):
    """m-major and n-major stores differ in descriptor and atom, never in arithmetic."""
    A, Wg, Wp, bg, bp, mk = _build(bias="both", mask=True)
    n_major = _run(
        A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, False), chunk_g=chunk_g, bg=bg, bp=bp, mask=mk
    )
    m_major = _run(
        A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, True), chunk_g=chunk_g, bg=bg, bp=bp, mask=mk
    )
    assert m_major.stride() == (1, _M), "the transposed output should be m-major"
    assert_bitwise(m_major, n_major, what="m-major vs n-major store")


@requires_sm90
@pytest.mark.parametrize(
    "schedule",
    [{"persistent": False}, {"persistent": True}, {"persistent": True, "pingpong": True}],
    ids=["one_shot", "persistent", "pingpong"],
)
@matrix_exempt(
    "sweeps the SCHEDULE combinations rather than a matrix axis: pingpong is not an independent "
    "axis because it requires persistent=True, so the legal settings are a three-element list, not "
    "a product. The persistent axis itself is swept by the matrix elsewhere"
)
def test_the_schedule_does_not_change_the_result(schedule):
    """Work-tile order is a scheduling choice; each tile still accumulates identically."""
    A, Wg, Wp, bg, bp, mk = _build(bias="both", mask=True)
    base = _run(A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, False), bg=bg, bp=bp, mask=mk)
    got = _run(
        A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, False), bg=bg, bp=bp, mask=mk, **schedule
    )
    assert_bitwise(got, base, what=f"{schedule} vs the default schedule")


@requires_sm90
@DUAL_GATED.parametrize(
    "cluster_n",
    only={"cluster_n": (1, 2)},
    because="both pool values are swept; the region that refuses cluster_n=2 needs chunk_g>1",
)
def test_a_cluster_along_n_does_not_change_the_result(cluster_n):
    """Multicasting B across a cluster changes who loads it, not what is computed."""
    A, Wg, Wp, bg, bp, mk = _build(bias="both", mask=True)
    base = _run(A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, False), bg=bg, bp=bp, mask=mk)
    got = _run(
        A,
        Wg,
        Wp,
        _out_for(_M, _N, torch.bfloat16, False),
        bg=bg,
        bp=bp,
        mask=mk,
        cluster_N=cluster_n,
    )
    assert_bitwise(got, base, what=f"cluster_N={cluster_n} vs cluster_N=1")


@requires_sm90
@matrix_exempt(
    "asserts the mask's SEMANTICS -- that it multiplies after the gate -- rather than sweeping "
    "configurations. One cell states it; more would restate it"
)
def test_the_mask_multiplies_after_the_gate_not_before():
    """``mask * sigmoid(g) * u``, not ``sigmoid(mask*g) * ...`` -- the gate is nonlinear.

    Pinned against the UNMASKED kernel output scaled on the host, so the comparison isolates the
    mask from every other source of error. Not bitwise: the kernel multiplies in fp32 and then
    rounds once, while the host rounds the kernel's already-rounded output, so the two differ by
    the double rounding -- one ulp, which is what the tolerance admits.
    """
    A, Wg, Wp, _, _, mk = _build(mask=True)
    unmasked = _run(A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, False))
    masked = _run(A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, False), mask=mk)
    expected = unmasked.float() * mk.float().reshape(-1, 1)
    assert_elementwise(
        masked.float(), expected, tolerance_bound(expected, 0.0, 8e-3), what="masked output"
    )


# ------------------------------------------------------------------ refusals


@requires_sm90
@DUAL_GATED.parametrize_unsupported(
    "ab_dtype", "chunk_g", "bias", "cluster_n", "gate3", "arg_fault"
)
def test_dual_gated_gemm_unsupported_combos_raise(
    ab_dtype, chunk_g, bias, cluster_n, gate3, arg_fault, expected_error, expected_match
):
    """Every declared unsupported combo is refused BY THIS KERNEL'S OWN front door.

    `pytest.raises` rather than `xfail` deliberately: an xfail'd correctness assertion cannot tell
    "refused" from "answered wrongly", so a kernel that silently computed garbage here would keep
    the suite green.
    """
    built = _build(dtype=ab_dtype, bias=bias, N3=_K if gate3 else 0)
    A, Wg, Wp, bg, bp, mk = built[:6]
    W3, b3 = (built[6], built[7]) if gate3 else (None, None)
    out = _out_for(_M, _N, ab_dtype, False)
    if arg_fault == "mask_dtype":
        mk = torch.rand(_M, device="cuda", dtype=torch.float64)
    elif arg_fault == "mask_extent":
        mk = torch.rand(_M + 1, device="cuda", dtype=ab_dtype)
    elif arg_fault == "bias_extent" and bias == "both":
        bg = torch.randn(_N + 1, device="cuda", dtype=torch.float32)
        bp = torch.randn(_N, device="cuda", dtype=torch.float32)
    elif arg_fault == "postact_extent":
        out = _out_for(_M, _N + 1, ab_dtype, False)
    with front_door_raises(expected_error, expected_match):
        _run(
            A,
            Wg,
            Wp,
            out,
            chunk_g=chunk_g,
            bg=bg,
            bp=bp,
            mask=mk,
            cluster_N=cluster_n,
            tile_N=2 * _N,
            W3=W3,
            b3=b3,
        )


@requires_sm90
@matrix_exempt("asserts a front-door contract that no matrix axis parametrizes")
def test_a_contraction_mismatch_is_named_rather_than_left_to_the_descriptor():
    """A K disagreement must name the extents, not surface as an FFI symbol mismatch."""
    A, Wg, Wp, *_ = _build(K=128)
    Wg2 = torch.randn(_N, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"Wg must be"):
        _run(A, Wg2, Wg2, _out_for(_M, _N, torch.bfloat16, False))
    with pytest.raises(ValueError, match=r"Wp must be"):
        _run(A, Wg, Wg2, _out_for(_M, _N, torch.bfloat16, False))


@requires_sm90
@matrix_exempt("asserts a front-door contract that no matrix axis parametrizes")
def test_the_dynamic_scheduler_refuses_to_run_without_its_semaphore():
    """Hopper has no cluster-launch-control fallback, so a missing counter is not recoverable."""
    A, Wg, Wp, *_ = _build()
    with pytest.raises(ValueError, match=r"requires tile_count_semaphore"):
        _run(
            A,
            Wg,
            Wp,
            _out_for(_M, _N, torch.bfloat16, False),
            persistent=True,
            is_dynamic_persistent=True,
        )


@requires_sm90
@matrix_exempt("asserts a front-door contract that no matrix axis parametrizes")
def test_an_unknown_gate_names_the_available_ones():
    """The set of gates is deliberately small, so listing it is the useful half of the message."""
    A, Wg, Wp, *_ = _build()
    with pytest.raises(ValueError, match=r"unknown gate activation 'swiglu'"):
        _run(A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, False), activation="swiglu")


# ------------------------------------------------------------------ structure and compile


@matrix_exempt("asserts the functor's parameter-pack contract; needs no GPU")
def test_chunk_g_is_a_compile_time_parameter_and_is_immutable():
    """`chunk_g` is folded into the kernel, so a post-construction write must raise.

    This is the concrete thing adopting `TemplateParams` bought: upstream set `chunk_g` by patching
    the attribute onto an already-built instance, which is reassignable and therefore able to
    desynchronize the functor from the kernel compiled from it.
    """
    assert issubclass(DualGatedGemmParams, GemmSm90Params)
    assert "chunk_g" in DualGatedGemmParams.field_names()
    assert GemmSm90Params.field_names() < DualGatedGemmParams.field_names(), (
        "the gate's pack must EXTEND the base's, not replace it -- `_bind_params` binds exactly one "
        "pack, so a pack missing the base fields would leave the functor half-configured"
    )
    gemm = DualGatedGemmSm90(cutlass.Float32, cutlass.BFloat16, (128, 128), (1, 1, 1), chunk_g=16)
    assert gemm.chunk_g == 16
    with pytest.raises(AttributeError, match=r"immutable after construction"):
        gemm.chunk_g = 1


@matrix_exempt("asserts the parameter pack covers both phases and nothing is left loose")
def test_the_pack_covers_both_phases_and_chunk_g_rides_the_construction_one():
    """`chunk_g` is a CONSTRUCTION parameter; the operand facts are CALL parameters. Both are bound.

    The split is not "compile-time vs runtime" -- ``__call__`` is ``@cute.jit``, so BOTH phases are
    folded into the kernel. It is "known at construction vs read off the operands", and the reason
    the distinction has to be expressed at all is that a functor cannot know its operand dtypes
    until it is handed tensors. What matters is that every ``self.X`` a traced method reads belongs
    to one phase or the other (or is derived from them), which is what makes the compile key
    complete instead of hand-maintained.

    ``chunk_g`` is on the construction side because the caller chooses it, and it selects the
    register pairing and the epilogue tile before any tensor is seen.
    """
    import cutlass

    gemm = DualGatedGemmSm90(cutlass.Float32, cutlass.BFloat16, (128, 128), (1, 1, 1), chunk_g=1)
    assert set(gemm.param_dict()) == set(DualGatedGemmParams.field_names())
    assert "chunk_g" in gemm.param_dict()
    # Before ``__call__`` the call phase is unbound, so the key's CALL half is empty -- but the key
    # is no longer the construction half alone. It also carries the post-construction
    # ``const_expr`` gates (``COMPILE_GATED_ATTRS``), which read ``"<unset>"`` on a functor nobody
    # configured. That third component is the fix for the 37-gate gap: `Params` freezes at binding
    # and every ``configure_a2a*`` writes AFTER construction, so a key equal to `param_dict()` was
    # exactly the key that could not tell two configurations apart.
    #
    # So the relation is CONTAINMENT, not equality, and the two halves of that are what this
    # asserts: every construction parameter is in the key with its value, and the surplus is
    # accounted for -- declared gates, all of them unset here, and nothing else.
    key = gemm.compile_key()
    assert gemm.param_dict().items() <= key.items(), (
        "a construction parameter went missing from the compile key"
    )
    from fold_cp_ops._internal.compile_time.template_params import is_key_component

    declared = set(type(gemm).compile_gated_attrs())
    surplus = set(key) - set(gemm.param_dict())
    # A DECLARED gate reads either the `"<unset>"` sentinel -- nothing has written it, which is a
    # distinct compile from any value it might later hold -- or a CLASS-LEVEL default, because
    # `getattr` finds one (`_gate3_full_width` is True here). Both are legitimate; what must not
    # happen is a value the key cannot carry.
    bad_declared = {
        n: type(key[n]).__name__
        for n in surplus & declared
        if key[n] != "<unset>" and not is_key_component(key[n])
    }
    assert not bad_declared, f"a declared gate resolved to something the key cannot carry: {bad_declared}"
    # The rest of the surplus is the `__dict__` SWEEP -- ordinary instance attributes holding a
    # compile-time value, e.g. `shared_storage=None`. That third source is the half nobody has to
    # maintain: `configure_a2a*` writes ~39 attributes on the A2A functor, and a hand-kept list is
    # wrong the first time a knob is added. Over-inclusion here is the SAFE direction -- a constant
    # that never varies costs nothing, and a value that does vary would have cost a wrong artifact.
    bad_swept = {
        n: type(getattr(gemm, n)).__name__
        for n in surplus - declared
        if not is_key_component(getattr(gemm, n))
    }
    assert not bad_swept, f"the key carries a value the sweep should have rejected: {bad_swept}"
    assert not hasattr(gemm, "rounding_mode"), (
        "an operand-phase parameter must not exist before __call__ binds it; a constructor default "
        "would be a second, stale source for a value folded into the kernel"
    )
    assert {"b_dtype", "a_layout", "rounding_mode", "epi_tile"} <= set(
        DualGatedGemmSm90.CallParams.field_names()
    )


@requires_sm90
@matrix_exempt("times a compile rather than sweeping configurations")
def test_dual_gated_gemm_cold_compile_under_five_seconds(tmp_path, monkeypatch):
    """The repo's hard compile bar, measured cold on the shape family this kernel ships for."""
    monkeypatch.setenv("CPO_CACHE_ENABLED", "0")
    A, Wg, Wp, bg, bp, mk = _build(bias="both", mask=True)
    out = _out_for(_M, _N, torch.bfloat16, False)
    t0 = time.perf_counter()
    _run(A, Wg, Wp, out, chunk_g=16, bg=bg, bp=bp, mask=mk)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, (
        f"cold compile took {elapsed:.1f}s, over the 5s bar. Check for a nested `if` inside a "
        f"`cutlass.range_constexpr` loop -- that is the known root-cause class."
    )


#: How each field of `DualGatedGemmSm90`'s two packs reaches `_compile_dual_gated_gemm`'s cache
#: key: the key parameter that distinguishes it, the key parameters it is DERIVED from, or None
#: when this front door pins it to a constant.
_PACK_FIELD_TO_KEY = {
    "acc_dtype": None,  # pinned: always Float32
    "fp8_fast_accum": None,  # pinned: the gate is 16-bit only
    "a_dtype": ("a_dtype",),
    "tile_shape_mn": ("tile_shape_mn",),
    "cluster_shape_mnk": ("cluster_shape_mnk",),
    "pingpong": ("pingpong",),
    "is_persistent": ("persistent",),
    "chunk_g": ("chunk_g",),
    "n_dual_tiles": ("n_dual_tiles",),
    "gate3_n3": ("gate3_n3",),
    "b_dtype": ("b_dtype",),
    "d_dtype": None,  # pinned: the gated kernel produces only a post-activation, so D is absent
    "c_dtype": None,  # pinned: no addend
    "a_layout": ("a_major",),
    "b_layout": ("b_major",),
    "d_layout": None,
    "c_layout": None,
    "rounding_mode": None,  # pinned: RN; the post-activation store refuses anything else on SM90
    "two_tensor_B": ("two_tensor_B",),
    # pinned: the dual-gated front door takes no second A operand -- the two-A x_gate path is a
    # different kernel with its own compile entry, which takes `a2_major`. So `a2_layout` is
    # always None through `dual_gated_gemm()` and cannot select another artifact.
    "a2_layout": None,
    "cta_tile_k": ("a_dtype",),
    "epi_tile": ("tile_shape_mn", "chunk_g", "postact_dtype"),
    "ab_stage": ("tile_shape_mn", "a_dtype", "b_dtype", "chunk_g"),
    "epi_stage": ("tile_shape_mn", "chunk_g", "postact_dtype"),
    "epi_c_stage": None,
}


@matrix_exempt("relates two declarations to each other; there is no kernel launch to parametrize")
def test_the_jit_cache_key_covers_the_functor_s_entire_compile_time_surface():
    """Every field of the gated functor's packs is keyed, derived from a keyed value, or pinned.

    Same contract as the plain GEMM's, and it earns a second copy here because the gate's pinned set
    is DIFFERENT: `d_dtype`/`c_dtype`/`rounding_mode` are constants at this front door (the kernel
    produces only a post-activation, takes no addend, and refuses anything but round-to-nearest),
    while `chunk_g` and `two_tensor_B` are keyed and are the two the plain GEMM does not have.

    A pack field that is neither keyed nor derived nor pinned means two configurations can share one
    compiled ``.o`` -- silently, across processes, through the persistent disk cache.
    """
    import inspect

    from fold_cp_ops.kernels.dual_gated_gemm import _compile_dual_gated_gemm

    key_params = set(inspect.signature(_compile_dual_gated_gemm.__wrapped__).parameters)
    surface = set(DualGatedGemmSm90.Params.field_names()) | set(
        DualGatedGemmSm90.CallParams.field_names()
    )

    unclassified = surface - set(_PACK_FIELD_TO_KEY)
    assert not unclassified, (
        f"these compile-time parameters are not classified against the cache key: "
        f"{sorted(unclassified)}. Add each to _PACK_FIELD_TO_KEY as keyed, derived or pinned."
    )
    stale = set(_PACK_FIELD_TO_KEY) - surface
    assert not stale, f"_PACK_FIELD_TO_KEY names fields that no longer exist: {sorted(stale)}"

    for field, sources in _PACK_FIELD_TO_KEY.items():
        if sources is None:
            continue
        missing = set(sources) - key_params
        assert not missing, (
            f"{field!r} is said to be keyed/derived via {sorted(missing)}, which "
            f"_compile_dual_gated_gemm does not take."
        )

    pinned = {f for f, s in _PACK_FIELD_TO_KEY.items() if s is None}
    reachable = set(inspect.signature(dual_gated_gemm).parameters)
    assert not (pinned & reachable), (
        f"{sorted(pinned & reachable)} are pinned in the cache key but SETTABLE through "
        f"dual_gated_gemm(); a caller varying one would reuse another configuration's kernel"
    )


@matrix_exempt("relates this tree's declared pool to main's; there is no kernel launch")
def test_the_tuning_pool_reproduces_mains_gated_configs():
    """One axis over {1, 16} -- `main`'s `_ggg_perf_configs`, including what it does NOT sweep.

    **The absent axes are the load-bearing part.** `main` pinned `tile_M` at 128 because 256
    measured 6-14x slower and 192 is invalid for this epilogue, and left `tile_N` to the caller. So
    a pool that added tiles here would be sweeping configurations `main` already measured and
    rejected -- paying compile and benchmark time per shape to rediscover a known answer, and
    widening the pool past the one this tree is being compared against.
    """
    from fold_cp_ops.kernels.dual_gated_gemm import DUAL_GATED_TUNING_SPACE

    assert [a.name for a in DUAL_GATED_TUNING_SPACE.axes] == ["chunk_g"], (
        "the gated kernel's only tuned knob is the weight layout; adding an axis here means "
        "re-litigating a measurement main already made"
    )
    assert {c["chunk_g"] for c in DUAL_GATED_TUNING_SPACE.configs()} == {1, 16}
    assert DUAL_GATED_TUNING_SPACE.excluded_count() == 0


@matrix_exempt("a property of the validity callback; the shapes are arguments to it")
@pytest.mark.parametrize(
    "K,N,expect",
    [
        (1024, 512, {1, 16}),  # both aligned -> both layouts admissible
        (1000, 512, {1}),  # K off the 16-floor -> block interleave dropped
        (1024, 500, {1}),  # N off the 16-floor -> block interleave dropped
        (7, 3, {1}),  # neither -> only the universal layout survives
    ],
)
def test_the_validity_rule_ports_mains_alignment_floor(K, N, expect):
    """`chunk_g=16` needs K and N both multiples of 16; `chunk_g=1` runs anything.

    This is `main`'s `_ggg_prune`, and it is a floor rather than a preference: the block-interleave
    layout pairs register `j` with register `j+H` across a 16-wide stmatrix atom, so an unaligned K
    or N does not make it slow, it makes the pairing wrong.

    The last row is the one that matters most -- the pool must NEVER empty. An empty pool leaves the
    kernel with nothing to run, and `chunk_g=1` is what guarantees it cannot happen.
    """
    from fold_cp_ops.kernels.dual_gated_gemm import (
        DUAL_GATED_TUNING_SPACE,
        dual_gated_config_is_valid,
    )

    request = {
        "A": torch.empty(8, K, device="meta"),
        "Wg": torch.empty(N, K, device="meta"),
    }
    kept = {
        c["chunk_g"]
        for c in DUAL_GATED_TUNING_SPACE.configs()
        if dual_gated_config_is_valid(c, request)
    }
    assert kept == expect
    assert kept, "the pool must never empty -- chunk_g=1 runs any shape the front door accepts"


@matrix_exempt("a property of the tuned wrapper's signature; no kernel launch")
def test_the_tuned_entry_refuses_a_pinned_chunk_g():
    """Pinning `chunk_g` on the tuned entry is refused, not merged.

    A caller pinning the one knob the tuner sweeps would make the measured winner and the executed
    kernel differ -- and the cache would then record a winner against a measurement that never
    happened, for every later call at that shape.
    """
    a = torch.zeros(8, 8, device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"tuned knobs and cannot also be passed"):
        dual_gated_gemm(a, a, a, a, 128, 128, chunk_g=16, do_autotune=True)


@matrix_exempt("audits the matrix itself")
def test_dual_gated_gemm_matrix_is_diverse():
    """Every facet of every axis is straddled by its pool, above the entropy floor.

    ``tests/testing/test_kernel_matrix.py`` runs this same check across all registered matrices;
    having it here too means a pool narrowed in THIS file fails in this file, where the fix is.
    """
    from fold_cp_ops.testing.kernel_matrix import MIN_FACET_ENTROPY

    weak = [
        s
        for axis in DUAL_GATED.axes
        for s in axis.diversity()
        if s.waived is None and s.entropy < MIN_FACET_ENTROPY
    ]
    assert not weak, "under-covered facets:\n  " + "\n  ".join(str(s) for s in weak)


@matrix_exempt("audits the matrix itself")
def test_every_supported_cell_is_outside_the_unsupported_regions():
    """The values the correctness tests actually run must not land in a region declared to raise."""
    import itertools

    supported = {
        "ab_dtype": (torch.bfloat16, torch.float16),
        "chunk_g": (1, 16),
        "bias": ("none", "both"),
        "cluster_n": (1,),
        "gate3": (False,),
        "arg_fault": ("none",),
    }
    for combo in itertools.product(*supported.values()):
        bound = dict(zip(supported, combo))
        for region in DUAL_GATED.regions():
            assert not region.where(**{k: bound[k] for k in region.axis_names()}), (
                f"{bound} is run as a supported cell but lands in the region {region.reason!r}"
            )


# ------------------------------------------------------------------ the fused output gate


@requires_sm90
@DUAL_GATED.parametrize(
    "bias",
    "mask",
    drop={"bias": ("gate_only",)},
    because="the half-biased cell is refused at the front door, so it has no correct output",
)
def test_the_output_gate_does_not_perturb_the_dual_output_it_shares_a_mainloop_with(bias, mask):
    """Adding the output gate must leave the dual result BIT-IDENTICAL.

    This is what makes the region-aware design's central claim testable: an output-gate work tile is
    an ORDINARY work tile of a WIDER operand, so the dual tiles must see exactly what they saw
    before. A design that instead squeezed the gate into the dual tiles' epilogue, or that changed
    the pipeline depth to make room, would perturb them -- slightly, plausibly, and invisibly to any
    tolerance test.
    """
    A, Wg, Wp, bg, bp, mk, W3, b3 = _build(bias=bias, mask=mask, N3=_K)
    gated, _ = _run(
        A,
        Wg,
        Wp,
        _out_for(_M, _N, torch.bfloat16, False),
        bg=bg,
        bp=bp,
        mask=mk,
        tile_N=2 * _N,
        W3=W3,
        b3=b3,
    )
    plain = _run(
        A, Wg, Wp, _out_for(_M, _N, torch.bfloat16, False), bg=bg, bp=bp, mask=mk, tile_N=2 * _N
    )
    assert_bitwise(gated, plain, what="dual output with vs without the output gate")


@requires_sm90
@pytest.mark.parametrize("N3", [128, 320, 64], ids=["exact_tile", "partial_tile", "narrow"])
@matrix_exempt(
    "sweeps the output gate's WIDTH, which is not a matrix axis: the axis is whether the gate "
    "exists, and its width is a shape. N3=320 against a 256-wide work tile is the "
    "partial-last-tile case, where the store's predicate is the only thing stopping an overrun"
)
def test_the_output_gate_matches_its_reference_including_a_partial_last_tile(N3):
    """The gate's own result is correct, and its partial last tile does not overrun.

    The padding rows appended after ``W3`` are zero-filled weights, so their accumulator is zero and
    the sigmoid of the bias would be a perfectly plausible value to store -- which is why the store
    is predicated to the true extent rather than left to produce harmless-looking numbers.

    Gated per element rather than against a scalar: the gate's output lives in ``(0, 1)``, where a
    max-error-over-max-reference ratio would accept almost anything.
    """
    A, Wg, Wp, bg, bp, mk, W3, b3 = _build(bias="both", N3=N3)
    _, gate3 = _run(
        A,
        Wg,
        Wp,
        _out_for(_M, _N, torch.bfloat16, False),
        bg=bg,
        bp=bp,
        tile_N=2 * _N,
        W3=W3,
        b3=b3,
    )
    _, ref3 = dual_gated_gemm_ref(A, Wg, Wp, bg, bp, W3=W3, b3=b3)
    assert gate3.shape == (_M, N3)
    assert_elementwise(gate3, ref3.double(), sigmoid_error_bound(A, W3, ref3, torch.bfloat16))


@requires_sm90
@matrix_exempt("checks a SHAPE precondition of the output gate, not a matrix axis")
def test_an_output_gate_whose_region_boundary_is_off_tile_is_refused():
    """``2N`` must fall on a work-tile edge, or the two regions would share a tile.

    A shared tile is the one case the runtime region test cannot express: the branch is per work
    tile, so a tile straddling the boundary would take one arm for columns belonging to the other.
    """
    A, Wg, Wp, bg, bp, mk, W3, b3 = _build(N=96, bias="both", N3=_K)
    out = _out_for(_M, 96, torch.bfloat16, False)
    with pytest.raises(ValueError, match=r"must be a multiple of tile_N"):
        _run(A, Wg, Wp, out, bg=bg, bp=bp, tile_N=128, W3=W3, b3=b3)


@requires_sm90
@matrix_exempt("checks the pairing of two arguments, which is not a matrix axis")
def test_the_output_gates_weight_and_destination_must_be_given_together():
    """One without the other is a caller error with no sensible interpretation."""
    A, Wg, Wp, bg, bp, mk, W3, b3 = _build(bias="both", N3=_K)
    out = _out_for(_M, _N, torch.bfloat16, False)
    with pytest.raises(ValueError, match=r"must be given together"):
        dual_gated_gemm(A, Wg, Wp, out, tile_M=128, tile_N=2 * _N, W3=W3, b3=b3, PostAct3=None)


# ------------------------------------------------------------------ the host operand build
#
# `chunk_g == 1` needs an interleaved weight, and there is no way to cache one inside a stateless
# entry point, so the build runs on every call. What it must NOT do is run as a chain of torch ops:
# each is a separate launch, and at the shapes this kernel is fast at, four launches of prep around
# a 7 us GEMM is most of the call. `test_the_operand_build_costs_one_device_launch` is the gate that
# keeps it fused; the bitwise tests around it are what make fusing it safe.


def _device_kernels(fn):
    """The device kernels one `fn()` call launches, memcpy and memset excluded.

    Purpose
        Turns "the operand build is one launch" into something a test can assert, rather than a
        claim that has to be re-measured by hand whenever the build changes.

    Semantics
        Calls `fn` once UNPROFILED first, so a cold compile, a `jit_cache` miss or a lazily created
        allocator block lands outside the measured window -- any of those would otherwise show up as
        extra launches and make the count irreproducible. Memcpy and memset are excluded because
        they are allocator traffic, not work the build chose to do.

    Args:
        fn: A zero-argument callable that performs exactly the operation under test. It must not
            allocate random data (`torch.randn` is itself a launch) -- build the inputs outside.

    Returns:
        A list of the device kernels' names, in launch order.
    """
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        e.key
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and "Memcpy" not in e.key
        and "Memset" not in e.key
    ]


@requires_sm90
@DUAL_GATED.parametrize(
    "ab_dtype",
    drop={"ab_dtype": (torch.float32,)},
    because=(
        "fp32 is the declared unsupported OPERAND dtype. The interleave itself accepts it -- it is "
        "a copy, so it is dtype-generic -- but a weight this helper would never be asked to "
        "prepare has nothing downstream to be identical for"
    ),
)
@pytest.mark.parametrize(
    "N,K", [(128, 128), (130, 100), (33, 17)], ids=["aligned", "off_grid", "odd"]
)
def test_the_fused_interleave_is_bitwise_its_torch_reference(ab_dtype, N, K):
    """The fused kernel and the torch chain produce the same bytes, at any extent.

    A copy is exact, so there is nothing here for a tolerance to absorb: if the fused kernel ever
    disagrees with `torch.stack(...).reshape(...)` it has mis-addressed a row, and the GEMM that
    reads the result would then contract the wrong weight against the right activation -- a
    plausible-looking wrong answer with no error anywhere.

    The off-grid and odd extents are the point of the shape axis: the kernel predicates its K tail
    rather than requiring a tile multiple, so an extent the reproduced kernel had to exclude
    (``N % 32``, ``K % 32``) must work here.
    """
    Wg = torch.randn(N, K, device="cuda", dtype=ab_dtype)
    Wp = torch.randn(N, K, device="cuda", dtype=ab_dtype)
    assert_bitwise(
        interleave_dual_weights(Wg, Wp),
        _interleave_dual_weights_torch(Wg, Wp),
        what=f"fused vs torch interleave at ({N}, {K}) {ab_dtype}",
    )


@requires_sm90
@matrix_exempt(
    "pins the host-side ORIENTATION of the interleaved weight, a property of the operand build "
    "rather than of any kernel configuration the matrix declares"
)
@pytest.mark.parametrize(
    "N,K",
    [(128, 128), (130, 100), (33, 17), (64, 128)],
    ids=["aligned", "off_grid", "odd", "square_2n_eq_k"],
)
def test_the_interleave_returns_k_by_2n_with_the_gate_on_even_columns(N, K):
    """Returns ``(K, 2N)``, gate on even COLUMNS -- pinned literally, not by comparison.

    **Why this exists when the fused-vs-torch tests already pass.** Every other test of this
    helper compares the fused kernel against `_interleave_dual_weights_torch`. Both halves
    define the same convention, so both would move together and every one of those tests would
    still pass: a self-consistent A/B cannot see the convention it is built on. That is not
    hypothetical. This package once diverged to the TRANSPOSE, ``(2N, K)``, while a front-A2A
    operand builder ported verbatim from the upstream kept the upstream's ``.mT``. The result
    was a GEMM N of ``2D`` where ``2*2D`` was intended -- half the N-tiles, and a peer store
    that never addressed the b-half of its destination -- with no exception raised anywhere,
    because the two shapes are merely different integers. The orientation has since been
    restored to the upstream's; this pins it so the divergence cannot return silently.

    **Why the ``2N == K`` cell is not redundant.** At that extent the two conventions have the
    SAME shape, so the shape assertion is blind and the column assertions are the only thing
    still holding the contract. Measured at ``(N, K) = (64, 128)``: the transposed orientation
    passes the shape check and fails the column check on 8126 of 8192 elements. A pool of
    non-square extents alone would have made a shape-only pin look sufficient.

    The column assertions additionally catch a gate/up SWAP, which has the right shape and the
    right columns as a set, and which no shape assertion can see.
    """
    Wg = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    Wp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    B = interleave_dual_weights(Wg, Wp)
    assert B.shape == (K, 2 * N), (
        f"the interleaved weight must be (K, 2N) = {(K, 2 * N)}; got {tuple(B.shape)}. Returning "
        "the transpose would make a consumer's `.mT` halve the GEMM's N extent instead of being "
        "the free view that hands the mainloop its n-major B operand."
    )
    assert B.stride(1) == 1, (
        f"(K, 2N) must be K-major, i.e. unit stride along 2N; got {B.stride()}. The stride IS the "
        "operand's major once `.mT` is taken -- `b_major` is a property of the physical layout, "
        "not of the view -- so a different major here is a different SMEM layout atom and a "
        "different cubin."
    )
    assert B.stride(0) % 8 == 0 and B.stride(0) >= 2 * N, (
        f"the row PITCH must be >= 2N and a multiple of 8 (16 B at 16-bit); got {B.stride(0)} for "
        f"2N={2 * N}. `.mT` turns this pitch into the n-major operand's leading stride, and a TMA "
        "descriptor requires that 16-B aligned -- padding the pitch is what keeps an arbitrary N "
        "legal instead of imposing N % 4 == 0."
    )
    assert_bitwise(B[:, 0::2], Wg.mT, what=f"even columns are the GATE projection at ({N}, {K})")
    assert_bitwise(B[:, 1::2], Wp.mT, what=f"odd columns are the UP projection at ({N}, {K})")


@matrix_exempt(
    "pins a pure-torch host-side LAYOUT contract; there is no kernel configuration to sweep and "
    "the function never touches a device"
)
@pytest.mark.parametrize(
    "twoN,K,N3,tile_N",
    [(256, 128, 128, 128), (256, 128, 129, 128), (256, 128, 8, 128), (512, 256, 96, 64)],
    ids=["n3_exact_tile", "n3_straddles", "n3_below_tile", "front_shape"],
)
def test_append_gate3_weight_lays_out_dual_then_gate_then_zero_pad(twoN, K, N3, tile_N):
    """``[dual 2N | W3 N3 | zeros]`` in a fresh k-major ``(2N + n3_pad, K)`` buffer.

    **Why this exists.** The function had no test at all, while being exported in ``__all__``,
    called from `build_dual_operands` and from the LayerNorm-fused front door, and depended on
    arithmetically by `fold_cp_ops.workflows.trimul_autotune`. Its two ``assert``s are INPUT
    checks: neither looks at the output, so every property below was unpinned. An assert on the
    way in is not coverage of what comes out.

    Each property is load-bearing, not cosmetic:

    * **The pad rounds UP to a whole ``tile_N``.** The dual/gate3 boundary is where the epilogue
      switches arms (``tile_coord_mnkl[1] >= n_dual_tiles``), so a short pad leaves the last gate
      tile reading a partial box.
    * **The pad rows are ZERO, not merely allocated.** They are contracted like any other row --
      the store predicates them away, but garbage there reaches a real tile's neighbours through
      the shared staging buffer.
    * **The dual region is byte-exact.** This is a copy; a tolerance would have nothing to absorb.

    ``n3_straddles`` (``N3 = tile_N + 1``) is the cell that separates a rounded pad from an
    unrounded one: the smallest ``N3`` that spills into a second tile, leaving ``tile_N - 1`` zero
    rows behind. Without it a ``n3_pad = n3`` implementation passes every other cell.
    """
    B = torch.randn(twoN, K, dtype=torch.bfloat16)
    W3 = torch.randn(N3, K, dtype=torch.bfloat16)
    n3_pad = (N3 + tile_N - 1) // tile_N * tile_N
    out = append_gate3_weight(B, W3, tile_N)
    assert out.shape == (twoN + n3_pad, K), (
        f"expected (2N + n3_pad, K) = {(twoN + n3_pad, K)}; got {tuple(out.shape)}. n3_pad must "
        f"round N3={N3} UP to a multiple of tile_N={tile_N}, or the last gate tile is partial."
    )
    assert out.stride() == (K, 1), (
        f"the combined weight must stay k-major (stride {(K, 1)}); got {out.stride()}. The stride "
        "is the operand's major, so a different one here is a different SMEM layout atom."
    )
    assert_bitwise(out[:twoN], B, what=f"dual rows preserved at 2N={twoN} K={K}")
    assert_bitwise(out[twoN : twoN + N3], W3, what=f"W3 rows land at [2N, 2N+N3) at N3={N3}")
    assert_bitwise(
        out[twoN + N3 :],
        torch.zeros_like(out[twoN + N3 :]),
        what=f"the {twoN + n3_pad - twoN - N3} pad rows are zero-filled, not merely allocated",
    )


@matrix_exempt("pins which input the guards reject and where they go blind; no kernel axis applies")
def test_append_gate3_weights_guards_refuse_a_misfit_but_go_blind_when_2n_equals_k():
    """The two guards cover for each other -- except at ``2N == K``, where both pass a transpose.

    **Why this is written down.** The presence of ``assert n2 % tile_N == 0`` invites the reader
    to treat the function as validating its operand's ORIENTATION. It does not. That assert reads
    ``B.shape[-2]``, which on a transposed ``(K, 2N)`` operand is ``K`` -- and at the front's
    ``K = 256, tile_N = 128`` that is still a clean multiple, so the guard passes on exactly the
    input it looks like it would catch.

    What actually saves the front shape is the OTHER assert: ``W3.shape[-1] == K`` reads ``2N``
    off the transposed operand and stops matching ``W3``'s real ``K``. The two therefore cover for
    each other -- until ``2N == K``, where a transposed operand has the SAME shape as a correct
    one, both asserts pass, and the function returns a buffer whose dual region is the transpose
    of the weight. Measured: at ``2N = K = 256`` it returns ``(384, 256)`` either way.

    The orientation guarantee does not live here. It lives in
    `test_the_interleave_returns_2n_by_k_with_the_gate_on_even_rows`, which pins the PRODUCER.
    This test exists so the divisibility assert is never read as a substitute for it.
    """
    tile_N = 128
    # 1. The divisibility guard does its own job: a dual region not ending on a work-tile edge.
    with pytest.raises(AssertionError, match="must be a multiple of tile_N"):
        append_gate3_weight(
            torch.randn(200, 128, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
            tile_N,
        )

    # 2. At the FRONT's shape the divisibility guard is blind to a transpose and the K guard is
    #    what refuses it. The first assert states the premise, so the test cannot pass vacuously.
    B = torch.randn(512, 256, dtype=torch.bfloat16)
    W3 = torch.randn(64, 256, dtype=torch.bfloat16)
    assert B.mT.shape[-2] % tile_N == 0, "premise: the divisibility guard PASSES on the transpose"
    with pytest.raises(AssertionError, match=r"W3 must be \(n3, k="):
        append_gate3_weight(B.mT.contiguous(), W3, tile_N)

    # 3. At 2N == K both guards pass, the transpose goes through, and only the CONTENT differs.
    B = torch.randn(256, 256, dtype=torch.bfloat16)
    W3 = torch.randn(64, 256, dtype=torch.bfloat16)
    good = append_gate3_weight(B, W3, tile_N)
    blind = append_gate3_weight(B.mT.contiguous(), W3, tile_N)
    assert blind.shape == good.shape == (384, 256), (
        f"premise: the two orientations are shape-indistinguishable here; got {tuple(good.shape)} "
        f"and {tuple(blind.shape)}"
    )
    assert_bitwise(
        blind[:256],
        B.mT,
        what="the blind path copied the TRANSPOSE byte for byte -- the guards saw nothing",
    )


@requires_sm90
@matrix_exempt(
    "the bias's dtype and the pitch pad are properties of the host build, not a kernel axis"
)
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("N", [128, 33], ids=["even", "odd"])
def test_the_interleaved_bias_is_widened_and_pitch_aligned(bias_dtype, N):
    """The kernel emits the bias as fp32 at a 4-element pitch, whatever the caller passed.

    Both properties are there to remove a launch, and both are easy to lose silently. Widening
    in-kernel is what lets a bf16 caller avoid two host casts -- and passing a 16-bit bias through
    instead measured 1.23x on the GEMM. The pitch pad is what lets an ODD N avoid a third launch
    inside `_pitch_align_broadcast`; without it the vector is correct but the call costs more.
    """
    Wg = torch.randn(N, 64, device="cuda", dtype=torch.bfloat16)
    Wp = torch.randn(N, 64, device="cuda", dtype=torch.bfloat16)
    bg = torch.randn(N, device="cuda", dtype=bias_dtype)
    bp = torch.randn(N, device="cuda", dtype=bias_dtype)
    _, bias = interleave_dual_weights(Wg, Wp, bg, bp, return_bias=True)
    assert bias.dtype is torch.float32, "the epilogue reads the bias as fp32; widening is in-kernel"
    assert bias.shape == (1, 2 * N)
    assert bias.stride(0) % 4 == 0, "the 4-element vectorized broadcast load needs a padded pitch"
    assert_bitwise(bias, _interleave_dual_bias_torch(bg, bp), what="fused vs torch bias interleave")


@requires_sm90
@DUAL_GATED.parametrize(
    "chunk_g", only={"chunk_g": (1, 16)}, because="8 is the declared unsupported value"
)
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@numeric_exempt(
    "asserts a LAUNCH COUNT from the profiler; the values are covered by the reference tests"
)
def test_the_operand_build_costs_one_device_launch(chunk_g, bias_dtype):
    """The whole ``chunk_g == 1`` operand build is ONE kernel; ``chunk_g > 1`` builds nothing.

    This is the gate on the reason the fused kernel exists. Built out of torch ops the element
    interleave costs a launch for the weight, one for the bias and one CAST PER BIAS VECTOR -- four
    for a 16-bit bias, against a GEMM that is 7 us at this shape. The count is asserted rather than
    the time because it is exact: a regression here is someone adding a `.float()` or a
    `.contiguous()`, which a timing test at a large shape would not resolve.

    ``chunk_g > 1`` is included as the contrast that makes the number meaningful: it loads both
    weights directly, so its only host work is widening a 16-bit bias, and it must launch nothing
    at all when the caller already passes fp32.
    """
    N, K = 128, 128
    Wg = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    Wp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    bg = torch.randn(N, device="cuda", dtype=bias_dtype)
    bp = torch.randn(N, device="cuda", dtype=bias_dtype)
    kernels = _device_kernels(
        lambda: build_dual_operands(Wg, Wp, bg, bp, chunk_g=chunk_g, tile_N=2 * N)
    )
    if chunk_g == 1:
        assert len(kernels) == 1, f"the element interleave must be one fused launch; got {kernels}"
        assert "interleave" in kernels[0], f"and it must be THE interleave kernel; got {kernels}"
    else:
        expect = 0 if bias_dtype is torch.float32 else 2
        assert len(kernels) == expect, (
            f"the block interleave builds no weight, so its only host work is the {expect} bias "
            f"widening cast(s); got {kernels}"
        )


@requires_sm90
@matrix_exempt("asserts the fallback contract of a host helper; the kernel never sees these inputs")
def test_the_interleave_falls_back_without_changing_the_answer():
    """A CPU or non-contiguous weight takes the torch path and yields the same values.

    The fused kernel assumes contiguity rather than checking it -- a k-strided source read as if
    contiguous is a wrong answer, not a fault -- so the guard in front of it is the whole safety
    argument. Both inputs here are ones a caller reaches by accident: a weight still on the host,
    and one that is a transposed view of another tensor.
    """
    N, K = 64, 32
    Wg = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    Wp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    assert_bitwise(
        interleave_dual_weights(Wg.cpu(), Wp.cpu()).cuda(),
        interleave_dual_weights(Wg, Wp),
        what="cpu fallback vs fused",
    )
    Wg_t = torch.randn(K, N, device="cuda", dtype=torch.bfloat16).T
    assert not Wg_t.is_contiguous(), "the point of this input is that it is not contiguous"
    assert_bitwise(
        interleave_dual_weights(Wg_t, Wp),
        _interleave_dual_weights_torch(Wg_t, Wp),
        what="non-contiguous fallback vs torch",
    )


# ------------------------------------------------------- the declared shape axes, exercised
#
# Three tests rather than one product: the audit requires every FACET of every axis to be reached
# by the module, not every combination, and a 6x6x5 product would be 180 launches to prove what 17
# prove. Each holds the other two extents at the workflow's own width so a failure names one axis.


@requires_sm90
@DUAL_GATED.parametrize("M")
def test_every_token_extent_is_accepted(M):
    """M is free: partial-tile, off-grid and multi-wave token counts all produce correct output.

    The multi-wave end is not a stress test -- it is where a persistent CTA visits more than one
    work tile, so anything the kernel carries across tiles is exercised for the first time.
    """
    A, Wg, Wp, bg, bp, mk = _build(M=M, N=_N, K=_K, bias="both", mask=True)
    out = _out_for(M, _N, torch.bfloat16, False)
    _run(A, Wg, Wp, out, chunk_g=1, bg=bg, bp=bp, mask=mk)
    ref = dual_gated_gemm_ref(A, Wg, Wp, bg, bp, mk).double()
    assert_elementwise(out, ref, gated_error_bound(A, Wg, Wp, ref, torch.bfloat16))


@requires_sm90
@DUAL_GATED.parametrize("N")
def test_every_feature_extent_is_accepted(N):
    """N is free, including the off-grid width and every TriMul feature width."""
    A, Wg, Wp, bg, bp, mk = _build(M=1000, N=N, K=_K, bias="both", mask=True)
    out = _out_for(1000, N, torch.bfloat16, False)
    _run(A, Wg, Wp, out, chunk_g=1, bg=bg, bp=bp, mask=mk)
    ref = dual_gated_gemm_ref(A, Wg, Wp, bg, bp, mk).double()
    assert_elementwise(out, ref, gated_error_bound(A, Wg, Wp, ref, torch.bfloat16))


@requires_sm90
@DUAL_GATED.parametrize("K")
def test_every_contraction_extent_is_accepted(K):
    """K is the one extent carrying a constraint -- the 16-byte floor -- and every pool value meets it."""
    A, Wg, Wp, bg, bp, mk = _build(M=1000, N=_N, K=K, bias="both", mask=True)
    out = _out_for(1000, _N, torch.bfloat16, False)
    _run(A, Wg, Wp, out, chunk_g=1, bg=bg, bp=bp, mask=mk)
    ref = dual_gated_gemm_ref(A, Wg, Wp, bg, bp, mk).double()
    assert_elementwise(out, ref, gated_error_bound(A, Wg, Wp, ref, torch.bfloat16))


@requires_sm90
@DUAL_GATED.parametrize("transpose_out", "persistent")
def test_the_store_majorness_and_the_schedule_are_both_free(transpose_out, persistent):
    """Both axes change transport only: same values, different descriptor and different tile order."""
    A, Wg, Wp, bg, bp, mk = _build(bias="both", mask=True)
    got = _run(
        A,
        Wg,
        Wp,
        _out_for(_M, _N, torch.bfloat16, transpose_out),
        chunk_g=1,
        bg=bg,
        bp=bp,
        mask=mk,
        persistent=persistent,
    )
    ref = dual_gated_gemm_ref(A, Wg, Wp, bg, bp, mk).double()
    assert_elementwise(got, ref, gated_error_bound(A, Wg, Wp, ref, torch.bfloat16))
