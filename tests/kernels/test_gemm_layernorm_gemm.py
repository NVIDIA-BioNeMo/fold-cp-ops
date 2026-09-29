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

"""Tests for ``fold_cp_ops/kernels/gemm_layernorm_gemm.py`` — ops 7, 8, 9 and 12 fused.

Three rungs, so a numerical failure names a stage rather than the chain:

* the token-pair einsum alone, against ``torch.einsum`` in fp32;
* its per-token statistics, against the fp32 moments of that same reference;
* the whole chain, against the fp32 oracle the kernel module ships.
"""

import pytest
import torch

import fold_cp_ops.kernels.gemm_layernorm_gemm as _glg
from fold_cp_ops.kernels.gemm_layernorm_gemm import (
    _CONT_PIPE_BY_ARCH,
    GemmLayerNormGemmSm90,
    _heuristic_cont_pipe,
    _token_pair_gemm_stats,
    gemm_layernorm_gemm,
    gemm_layernorm_gemm_ref,
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
from fold_cp_ops.testing.numerics import assert_bitwise, assert_elementwise

# ── the declared test matrix for this kernel ───────────────────────────────────────────────────
# **Add a shape HERE, not to one test.** Every test below draws on this; narrowing it requires a
# written `because=`. Enforced at collection by tests/conftest.py; see
# fold_cp_ops/testing/kernel_matrix.py for the mechanics.
GEMM_LAYERNORM_GEMM = KernelMatrix(
    kernel="gemm_layernorm_gemm",
    axes=(
        Axis(
            name="N",
            domain=(
                "the token extent (the triangle side); any positive int with N % 8 == 0, which is "
                "the 16-byte TMA alignment floor for a 16-bit dtype and the ONLY constraint. NOT "
                "restricted to powers of two, to a tile multiple, or to any grid alignment: a "
                "partial token tile is launched by a ceil-div grid, zero-filled by the TMA, and "
                "dropped by a compile-time-gated bounds check on the two stores"
            ),
            values=(
                # below one (blk_i, blk_j) tile -- the whole grid is a single partial tile
                72,
                128,
                # off-grid: N % 8 but not % 64, i.e. the predicated arm
                136,
                160,
                192,
                200,
                248,
                # tile multiples around the cont_pipe crossover at 640
                256,
                320,
                384,
                512,
                640,
                768,
                1024,
                # the A2A-fused TriMul token ladder; the top two only run when memory allows
                2048,
                4096,
                8192,
                12288,
            ),
            # tile=64 is (blk_i, blk_j); big=640 is the bisected cont_pipe crossover; small=128 is
            # the largest N whose grid is one token tile per axis.
            facets={
                **int_facets(tile=64, big=640, small=128),
                # This kernel FUSES op 7, so its N is the BACK half's token extent and the
                # series that applies is the N_token ladder itself -- NOT the projection
                # kernels' M = N_token**2 / cp. The values were already here; the facet is
                # what stops them being narrowed away unnoticed.
                "workflow_N_token": lambda v: v in (2048, 4096, 8192, 12288),
            },
            waived={
                "odd": (
                    "unreachable by construction: N % 8 == 0 is the 16-byte alignment floor, so no "
                    "odd N is in the supported domain at all. The refusal is covered by the "
                    "'N_not_mult8' arg_fault instead of by a pool value that could never compute"
                )
            },
        ),
        Axis(
            name="D",
            domain=(
                "the feature extent; any positive int with D % 8 == 0 (16-byte alignment for a "
                "16-bit dtype). D % 16 takes the unpredicated projection path; D % 8-not-16 has no "
                "%16 divisor, so the output tile drops to 16 with a ceil-div tile count and the "
                "affine vectors are zero-padded to the contraction tile"
            ),
            values=(
                # D % 8 but NOT % 16 -- the padded projection path
                72,
                136,
                200,
                264,
                # D % 16 but not % 64
                80,
                144,
                208,
                320,
                # the TriMul feature dims
                128,
                192,
                256,
                384,
                512,
            ),
            # tile=16 is the projection's output-tile granularity, which is what the padded path
            # turns on; big=256 is where the output tile stops growing; small=128 is the smallest
            # TriMul feature dim.
            facets={
                **int_facets(tile=16, big=256, small=128),
                # The four TriMul feature widths, under the table's name.
                "workflow_D": lambda v: v in (128, 256, 384, 512),
            },
            waived={
                "odd": (
                    "unreachable by construction: D % 8 == 0 is the 16-byte alignment floor. The "
                    "refusal is covered by the 'D_not_mult8' arg_fault"
                )
            },
        ),
        Axis(
            name="B",
            domain=(
                "batch; any positive int. It is folded onto the einsum's L axis rather than given "
                "its own grid dimension, so it costs no extra kernel -- but that folding is exactly "
                "what a cross-batch indexing slip would corrupt silently"
            ),
            values=(1, 2, 3),
            facets={
                "unbatched": lambda v: v == 1,
                "batched": lambda v: v > 1,
                "odd_batch": lambda v: v % 2 == 1,
            },
        ),
        Axis(
            name="direction",
            domain=(
                "which token axis the einsum contracts. 'outgoing' contracts the CONTIGUOUS one and "
                "'incoming' the strided one; the second is not a second kernel but the same one fed "
                "a transposed strided view, which flips both operands from k-major to mn-major"
            ),
            values=("outgoing", "incoming"),
            facets={
                "outgoing": lambda v: v == "outgoing",
                "incoming": lambda v: v == "incoming",
            },
        ),
        Axis(
            name="cont_pipe",
            domain=(
                "the token-pair GEMM's schedule: False is one k-pipeline per feature, True is one "
                "continuous pipeline across all (feature, k) tiles, None resolves it from N by the "
                "size heuristic. Both schedules are valid at EVERY shape -- the knob carries no "
                "size constraint -- so a wrong pick costs speed and never correctness"
            ),
            values=(None, False, True),
            facets={
                "heuristic": lambda v: v is None,
                "per_feature": lambda v: v is False,
                "continuous": lambda v: v is True,
            },
        ),
        Axis(
            name="has_bias",
            domain="the output projection's bias is optional",
            values=(False, True),
            facets={"with_bias": lambda v: v, "without_bias": lambda v: not v},
        ),
        Axis(
            name="has_norm_bias",
            domain=(
                "the LayerNorm's OWN bias (op 8's shift) is optional too, and separately from the "
                "projection's: this one enters inside the normalize prologue, that one in the "
                "epilogue. Absent, a zero is staged into the affine scratch -- the same value the "
                "contraction-tile padding already writes -- so the normalize adds the additive "
                "identity and no arithmetic changes. The two builds are DIFFERENT compiled kernels "
                "and must be keyed apart"
            ),
            values=(False, True),
            facets={"with_norm_bias": lambda v: v, "without_norm_bias": lambda v: not v},
        ),
        Axis(
            name="has_gate3",
            domain=(
                "op 12's elementwise output gate. Absent, the load and multiply are pruned by a "
                "const_expr gate, so the epilogue is identical to a build compiled without the "
                "feature -- which is precisely why the ABSENT case needs its own coverage"
            ),
            values=(False, True),
            facets={"gated": lambda v: v, "ungated": lambda v: not v},
        ),
        Axis(
            name="input_dtype",
            domain=(
                "bfloat16 / float16 -- the 16-bit types the SM90 WGMMA atom this chain builds "
                "accepts. float32 has no such atom, so it is not in the domain at all"
            ),
            values=(torch.bfloat16, torch.float16),
            facets=dtype_facets((torch.bfloat16, torch.float16)),
        ),
        Axis(
            name="arg_fault",
            domain=(
                "a CALLER MISTAKE, or 'none'. The values name something the caller got wrong rather "
                "than a kernel configuration, and every non-'none' one must be refused at this "
                "kernel's own front door with an API-level error. The pool spans BOTH kinds of "
                "mistake this chain can be handed: a shape that breaches the 16-byte alignment "
                "floor, and a tensor argument of the wrong dtype or extent. The extent faults earn "
                "their place by being unobservable otherwise -- the affine vectors are staged by a "
                "loop bounded by D, so a short one is read past its end and scales the tail of "
                "every row with unrelated memory, with no fault and no wrong-looking output"
            ),
            values=(
                "none",
                "N_not_mult8",
                "D_not_mult8",
                "bad_direction",
                "a_dtype",
                "proj_w_dtype",
                "norm_w_extent",
                "proj_w_extent",
                "gate3_extent",
                "b_dtype",
            ),
            facets={
                "well_formed": lambda v: v == "none",
                "bad_dtype": lambda v: v.endswith("_dtype"),
                "bad_extent": lambda v: v.endswith("_extent"),
                "alignment_floor": lambda v: v.endswith("_not_mult8"),
            },
        ),
    ),
    # Every region is keyed on `arg_fault` alone, deliberately: none of them is a property of a
    # SHAPE the kernel supports, so keying them on N or D would put a value in those pools that can
    # never compute and would have to be dropped, with a `because=`, by every correctness test that
    # sweeps them. Keeping the shape pools entirely inside the supported domain is what lets a
    # reader see the FIRST PRINCIPLE claim -- 16-byte alignment and nothing else -- in the pool.
    # The token-pair einsum, its per-token statistics, and the projection that follows --
    # a contraction feeding a row reduction feeding another contraction.
    computes=("contraction", "row_reduction", "saturating_activation"),
    property_waivers={
        # Waived on STRUCTURE, not on a sweep, and the distinction matters: the row this kernel
        # normalizes is the token-pair einsum's OUTPUT, `tri[b, i, j, :]`, not an input. Its mean
        # over the feature axis is whatever `sum_k a[b,d,i,k] * b[b,d,j,k]` comes to; no caller-side
        # knob sets it, so there is no `row_mean` to sample. Confirmed by trying: offsetting `a`
        # and `b` by 100 drives the reference itself to NaN, which measures the probe rather than
        # the kernel. The property that DOES apply here -- the einsum operands' distribution -- is
        # varied by the shape and dtype axes.
        "row_mean": (
            "measured not applicable: the normalized row is the einsum's OUTPUT, so no input sets "
            "its mean; offsetting the operands drives the fp32 reference to NaN"
        ),
    },
    unsupported=(
        Unsupported(
            where=lambda arg_fault: arg_fault == "N_not_mult8",
            raises=ValueError,
            match=r"N must be a multiple of 8",
            reason=(
                "N below the 16-byte TMA alignment floor. Nothing downstream refuses it: the TMA "
                "descriptor build is where it eventually fails, three frames in and naming a box "
                "extent rather than the argument the caller passed"
            ),
        ),
        Unsupported(
            where=lambda arg_fault: arg_fault == "D_not_mult8",
            raises=ValueError,
            match=r"D must be a multiple of 8",
            reason=(
                "D below the 16-byte alignment floor. The projection's tile resolution falls back "
                "to 16 for any D % 8, so a D that is 4 mod 8 resolves to a tile that does not "
                "cover it and the last columns are simply never written"
            ),
        ),
        Unsupported(
            where=lambda arg_fault: arg_fault == "bad_direction",
            raises=ValueError,
            match=r"direction must be 'outgoing' or 'incoming'",
            reason=(
                "the two directions contract DIFFERENT token axes, so there is no safe default. "
                "Silently falling back to one would return a fully-formed, plausible, wrong answer"
            ),
        ),
        Unsupported(
            where=lambda arg_fault: arg_fault == "a_dtype",
            raises=ValueError,
            match=r"activation dtype must be bfloat16 or float16",
            reason=(
                "float32 activations. The `input_dtype` axis states that the domain is the two "
                "16-bit types, and this is what OBLIGES that statement to be true: both GEMMs are "
                "SM90 WGMMA and its f16/bf16 MMA atom has no 32-bit form, so an fp32 activation has "
                "no kernel at all. Declared as a region rather than left implicit because a domain "
                "narrower than the pool is a claim nothing checks -- before the front-door guard, "
                "fp32 died inside MMA construction naming an MLIR type, which is exactly the "
                "'AttributeError three frames into a dispatch' the front-door rule exists to stop"
            ),
        ),
        Unsupported(
            where=lambda arg_fault: arg_fault == "proj_w_dtype",
            raises=ValueError,
            match=r"proj_w must already be the activation dtype",
            reason=(
                "the projection weight is a GEMM operand and there is no internal cast: casting it "
                "per call would repeat, on every call, a conversion the caller should do once. "
                "Without the front-door check it reaches the launch as a DSL type error naming an "
                "MLIR type rather than an argument"
            ),
        ),
        Unsupported(
            where=lambda arg_fault: arg_fault == "b_dtype",
            raises=ValueError,
            match=r"a and b must share an element type",
            reason=(
                "`a`'s dtype is validated on its own (the WGMMA atom has no 32-bit form) and the "
                "SHAPE of the (a, b) pair is validated one line above -- but nothing compared "
                "their DTYPES, so a `b` of a different element type passed both and reached the "
                "TVM-FFI ABI check as `Mismatched Tensor on argument #1`, naming an FFI slot "
                "rather than the argument. Both tensors are individually well formed, which is "
                "the point: this is the partner check, and it is the half that went missing "
                "because it was not kept adjacent to the shape comparison it belongs with"
            ),
        ),
        Unsupported(
            where=lambda arg_fault: arg_fault == "norm_w_extent",
            raises=ValueError,
            match=r"norm_w must have shape",
            reason=(
                "the LayerNorm gain is staged into shared memory by a loop bounded by D, with no "
                "bound of its own. A shorter one is read past its end, so the tail of every row is "
                "scaled by whatever follows it in memory -- a wrong answer with no fault raised"
            ),
        ),
        Unsupported(
            where=lambda arg_fault: arg_fault == "proj_w_extent",
            raises=ValueError,
            match=r"proj_w must have shape",
            reason=(
                "a projection weight of the wrong extent builds a TMA box that walks off the "
                "allocation, or -- worse when it is too LONG -- silently contracts against the "
                "wrong columns"
            ),
        ),
        Unsupported(
            where=lambda arg_fault: arg_fault == "gate3_extent",
            raises=ValueError,
            match=r"gate3 must hold",
            reason=(
                "the gate is reshaped to (M, D), and torch's reshape accepts any element count "
                "that factors -- so a gate built with the wrong token count re-indexes against the "
                "wrong tokens instead of failing"
            ),
        ),
    ),
)


def _inputs(
    B, N, D, dtype=torch.bfloat16, has_bias=True, has_gate3=False, seed=0, has_norm_bias=True
):
    """Build one well-formed input set for the fused chain.

    Purpose:
        One builder for every test here, so a shape that fails in one test is reproducible in
        another without re-deriving how its operands were made.

    Semantics:
        The activations are scaled by 0.1 so that the token-pair einsum, which sums ``N`` products,
        stays inside the 16-bit range at the largest ``N`` in the pool; without that scaling the
        reference itself would differ from the kernel by overflow rather than by rounding. The
        projection weight is scaled by ``D ** -0.5`` for the same reason one axis further on. The
        gate is drawn from ``rand`` rather than ``randn`` because op 11 makes it a sigmoid output,
        so its true range is ``[0, 1)`` and a normal draw would exaggerate the error it amplifies.

    Args:
        B: Batch, any positive int.
        N: Token extent. Must be a multiple of 8, or the front door refuses the call this feeds.
        D: Feature extent. Must be a multiple of 8, same reason.
        dtype: The activation dtype; must be one the WGMMA atom accepts (bfloat16 / float16).
        has_bias: Whether to build a projection bias.
        has_gate3: Whether to build an output gate.
        seed: The generator seed, so two calls with the same arguments give the same bytes.
        has_norm_bias: Whether to build the LayerNorm's OWN bias. False omits the ``norm_b`` key
            entirely, so ``_run`` passes None and the kernel compiles its bias-free build. Distinct
            from ``has_bias``, which is the PROJECTION bias in the epilogue.

    Returns:
        A dict with keys ``a``, ``b``, ``norm_w``, ``proj_w``, and optionally ``norm_b``,
        ``proj_b`` and ``gate3``. The projection weight is ALREADY in ``dtype``, which the front
        door requires.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = dict(
        a=torch.randn(B, D, N, N, device="cuda", dtype=dtype, generator=g) * 0.1,
        b=torch.randn(B, D, N, N, device="cuda", dtype=dtype, generator=g) * 0.1,
        norm_w=torch.randn(D, device="cuda", dtype=torch.float32, generator=g),
        proj_w=(torch.randn(D, D, device="cuda", dtype=torch.float32, generator=g) * (D**-0.5)).to(
            dtype
        ),
    )
    if has_norm_bias:
        out["norm_b"] = torch.randn(D, device="cuda", dtype=torch.float32, generator=g)
    if has_bias:
        out["proj_b"] = torch.randn(D, device="cuda", dtype=torch.float32, generator=g)
    if has_gate3:
        # `rand`, NOT `randn`, and the choice is load-bearing for more than realism. Every oracle
        # comparison here scores `max|diff| / std(ref)`, a MAX over B*N^2*D elements divided by an
        # RMS -- and a multiplicative gate inflates that ratio by roughly `max|gate3| / std(gate3)`,
        # because the numerator lands wherever the gate is largest while the denominator averages.
        # Uniform [0, 1) has a max/std of about 3.46; a normal draw has 5.2-6.0 at these sizes.
        # MEASURED with a randn gate on four of the cells below: ungated relerr 1.84e-2 to 2.35e-2,
        # gated 5.24e-2 to 5.29e-2 -- an amplification of 2.2-2.9x that puts every one of them just
        # over the 5e-2 bound, for a reason that is the input distribution and not the kernel.
        # Switching this to `randn` therefore breaks the oracle tests without breaking anything real.
        # `main` avoided the whole question by keeping gate3 out of its oracle comparisons and
        # checking it byte-exactly instead (see test_batch_slices_match_unbatched, which does that
        # here). Op 11 makes the gate a sigmoid output, so [0, 1) is also simply its true range.
        out["gate3"] = torch.rand(B * N * N, D, device="cuda", dtype=dtype, generator=g)
    return out


def _run(t, **kw):
    """Call the kernel with an input dict from :func:`_inputs`.

    Args:
        t: The dict :func:`_inputs` returned. ``proj_b`` and ``gate3`` are optional keys.
        **kw: Forwarded to ``gemm_layernorm_gemm`` (``direction``, ``eps``, ``cont_pipe``, ...).

    Returns:
        The ``(B, N, N, D)`` kernel output.
    """
    return gemm_layernorm_gemm(
        t["a"],
        t["b"],
        t["norm_w"],
        t.get("norm_b"),
        t["proj_w"],
        t.get("proj_b"),
        gate3=t.get("gate3"),
        **kw,
    )


def _spread_bound(ref, frac):
    """Per-element allowance ``frac * std(ref)`` -- this file's bar, as a bound rather than a metric.

    Purpose:
        This replaces a ``_relerr`` helper that returned ``max|got-ref| / std(ref)`` for the caller
        to compare against a constant. That test is algebraically the same one -- ``max_i e_i <= T``
        is ``for all i: e_i <= T`` for a scalar T -- but it reported a single number, so a failure
        said how big the worst error was and nothing about WHERE. Passing the same T to
        ``assert_elementwise`` keeps the bar identical and adds the offending index, both values and
        the violating count, which is what localizes a tile-edge or padded-lane defect.

        The bound is still scaled by the WHOLE tensor's spread, which is the honest description of
        what this file has always required: an element whose own magnitude is small is judged
        against the tensor's slack. Tightening that to a per-element relative term is a separate,
        MEASURED change -- making it inside a mechanical conversion would hide a behaviour change
        in a diff that looks like a rename.

    Args:
        ref: The fp32 reference. Only its standard deviation is read. Must not be constant: a zero
            spread yields a zero bound, which then demands an exact match and would be a surprising
            way to fail.
        frac: The fraction of the spread each element may deviate by. This file's values are 5e-2
            for a full fused output, 2e-2 for the bare einsum, and 5e-3 / 1e-2 for the statistics.

    Returns:
        A 0-d fp32 tensor, broadcastable against ``ref`` as ``assert_elementwise``'s ``bound``.
    """
    return frac * ref.float().std()


def _skip_if_oom(exc, B, N, D):
    """Turn a device-memory exhaustion into a skip, and anything else back into a failure.

    Purpose:
        The largest cells in the pool are genuinely supported and genuinely do not fit on every
        card -- the einsum output alone is ``B * N^2 * D`` elements. A memory-ESTIMATE gate would
        be a shape restriction in disguise and would drift out of date; catching the allocator's
        own refusal is the only form of this that cannot be wrong about what fits.

    Args:
        exc: The caught exception.
        B, N, D: The cell, for the skip message.

    Returns:
        None. Calls ``pytest.skip`` when ``exc`` is an out-of-memory error.

    Raises:
        The original exception, unchanged, when it is anything else.
    """
    if isinstance(exc, torch.OutOfMemoryError):
        torch.cuda.empty_cache()
        pytest.skip(f"B={B} N={N} D={D} does not fit in device memory ({exc.__class__.__name__})")
    raise exc


@GEMM_LAYERNORM_GEMM.parametrize(
    "N",
    "D",
    "arg_fault",
    only={
        "N": (128, 136, 200, 256, 320, 512, 768),
        "D": (72, 128, 144, 208, 256, 264, 384),
        "arg_fault": ("none",),
    },
    because=(
        "the broad end-to-end correctness grid, so N and D are the ladder-spanning subsets: N "
        "crosses the single-tile rung (128), the predicated off-grid arm (136, 200), the tile "
        "multiples, and both sides of the cont_pipe crossover at 640; D crosses the padded "
        "projection path (72, 264), the %16-not-%64 rung (144, 208) and the TriMul feature dims. "
        "The extremes -- N at the A2A token ladder, D at 512 -- have dedicated tests below, where "
        "they are not multiplied by the other axis. arg_fault is pinned to 'none' because this is "
        "a supported-path test; the faults are swept by test_arg_fault_raises."
    ),
)
def test_end_to_end(N, D, arg_fault):
    """The fused chain against the fp32 oracle, across the shape grid.

    The tolerance is relative to the reference's spread, not absolute, and it is loose (5e-2)
    on purpose: the kernel NARROWS the einsum result to 16 bits before the projection reads it,
    which the oracle does not, so at bfloat16 the two agree only to about 2e-2. Tightening this
    would be measuring the dtype, not the kernel.
    """
    assert arg_fault == "none"
    t = _inputs(1, N, D, seed=N * 31 + D)
    out = _run(t, eps=1e-5)
    ref = gemm_layernorm_gemm_ref(
        t["a"], t["b"], t["norm_w"], t.get("norm_b"), t["proj_w"], t["proj_b"], eps=1e-5
    )
    assert tuple(out.shape) == (1, N, N, D)
    assert_elementwise(out, ref, _spread_bound(ref, 5e-2), what=f"N={N} D={D}")


@GEMM_LAYERNORM_GEMM.parametrize(
    "N",
    "D",
    "input_dtype",
    "has_bias",
    "has_gate3",
    only={"N": (72, 192), "D": (80, 200), "input_dtype": (torch.bfloat16, torch.float16)},
    because=(
        "the FUNCTIONAL cross product -- dtype x bias x gate -- which is 16 cells per shape, so it "
        "runs on two small shapes rather than the whole ladder. One is off-grid in N and %16 in D "
        "(72, 80); the other is off-grid in N and %8-not-16 in D (192, 200), so both predication "
        "paths of the projection are crossed with every functional combination. The shape ladder "
        "itself is test_end_to_end's job."
    ),
)
def test_functional_variants(N, D, input_dtype, has_bias, has_gate3):
    """Every combination of dtype, projection bias and output gate, against the fp32 oracle.

    The gate is the one that needs the ungated cell as much as the gated one: absent, its load and
    multiply are pruned by a ``const_expr``, so the two builds are different kernels and a defect
    in either is invisible from the other.
    """
    t = _inputs(1, N, D, dtype=input_dtype, has_bias=has_bias, has_gate3=has_gate3, seed=7)
    out = _run(t, eps=1e-5)
    ref = gemm_layernorm_gemm_ref(
        t["a"],
        t["b"],
        t["norm_w"],
        t.get("norm_b"),
        t["proj_w"],
        t.get("proj_b"),
        gate3=t.get("gate3"),
        eps=1e-5,
    )
    assert out.dtype == input_dtype
    assert_elementwise(
        out,
        ref,
        _spread_bound(ref, 5e-2),
        what=f"N={N} D={D} {input_dtype} bias={has_bias} gate3={has_gate3}",
    )


@GEMM_LAYERNORM_GEMM.parametrize(
    "B",
    "direction",
    "N",
    "D",
    only={"N": (128, 160), "D": (128, 256)},
    because=(
        "the batch x direction cross product, which is 12 cells per shape pair. The shapes are one "
        "tile-aligned and one off-grid N against two TriMul feature dims; the batch folding is a "
        "property of the L-axis indexing, not of the tile geometry, so widening the shapes here "
        "would re-test test_end_to_end's grid at four times the cost."
    ),
)
def test_batched_directions(B, direction, N, D):
    """Both contraction directions at every batch, against the fp32 oracle."""
    t = _inputs(B, N, D, seed=B * 101 + N)
    out = _run(t, direction=direction, eps=1e-5)
    ref = gemm_layernorm_gemm_ref(
        t["a"],
        t["b"],
        t["norm_w"],
        t.get("norm_b"),
        t["proj_w"],
        t["proj_b"],
        direction=direction,
        eps=1e-5,
    )
    assert tuple(out.shape) == (B, N, N, D)
    assert_elementwise(out, ref, _spread_bound(ref, 5e-2), what=f"B={B} {direction} N={N} D={D}")


@GEMM_LAYERNORM_GEMM.parametrize(
    "B",
    "direction",
    "has_gate3",
    "has_bias",
    only={"B": (2, 3)},
    because=(
        "B=1 is excluded because this test's claim -- 'slice b of the batched output equals a "
        "standalone B=1 call on slice b' -- is vacuous at B=1. The shape is fixed at one off-grid "
        "cell: what is under test is the batch folding onto the einsum's L axis, which the tile "
        "geometry does not participate in."
    ),
)
def test_batch_slices_match_unbatched(B, direction, has_gate3, has_bias):
    """Each batch slice must be BIT-identical to a standalone unbatched call on that slice.

    This is the definitive check that the batch fold carries no cross-batch leak. A tolerance-based
    comparison against the oracle cannot make it: a leak between two slices of the same
    distribution moves the result by far less than the bf16 rounding the tolerance already admits.
    """
    N, D = 160, 128
    t = _inputs(B, N, D, has_bias=has_bias, has_gate3=has_gate3, seed=100 + B)
    out = _run(t, direction=direction, eps=1e-5)
    gate = t["gate3"].reshape(B, N * N, D) if has_gate3 else None
    for i in range(B):
        one = dict(t, a=t["a"][i : i + 1], b=t["b"][i : i + 1])
        if has_gate3:
            one["gate3"] = gate[i]
        ref = _run(one, direction=direction, eps=1e-5)
        assert_bitwise(
            out[i],
            ref[0],
            what=f"batch {i} of B={B} ({direction}, gate3={has_gate3}, bias={has_bias}) differs from the standalone call -- a cross-batch leak or a wrong token-axis mapping",
        )


@GEMM_LAYERNORM_GEMM.parametrize(
    "cont_pipe",
    "N",
    "D",
    only={"N": (128, 248, 512, 1024), "D": (128, 264)},
    because=(
        "the schedule knob is a PURE perf pick -- both values compute the same numbers -- so what "
        "this needs is N values on both sides of the heuristic's crossover at 640 rather than the "
        "whole ladder, crossed with one padded and one unpadded D to confirm the schedule and the "
        "projection's predication are independent."
    ),
)
def test_schedule_knob_agrees(cont_pipe, N, D):
    """The two token-pair schedules, and the heuristic's own pick, must all agree BIT-exactly.

    ``cont_pipe`` reorders loads and resets the accumulator at a different point; it changes no
    arithmetic. Anything but exact equality here means one of the two schedules drops or duplicates
    a contribution -- which a tolerance against the oracle would absorb.
    """
    t = _inputs(1, N, D, has_gate3=True, seed=N + D)
    got = _run(t, eps=1e-5, cont_pipe=cont_pipe)
    ref = _run(t, eps=1e-5, cont_pipe=False)
    assert_bitwise(
        got,
        ref,
        what=f"cont_pipe={cont_pipe} differs from the per-feature schedule at N={N} D={D}; the knob is a scheduling choice and must not change the result",
    )


@GEMM_LAYERNORM_GEMM.parametrize(
    "N",
    "D",
    only={"N": (2048, 4096, 8192, 12288), "D": (128, 512)},
    because=(
        "the A2A-fused TriMul token ladder, which this kernel is NOT selected for (its heuristic "
        "corner is N^2*D <= 8.4e6) but must still compute -- a heuristic change must not walk into "
        "an untested path. The cells are O(N^2*D) in memory, so the top of the ladder does not fit "
        "on every card and is skipped by the allocator's own refusal; the functional axes are "
        "pinned because this test is about SIZE."
    ),
)
def test_token_ladder_spot_check(N, D):
    """The A2A-fused TriMul token ladder, one shape at a time, against the fp32 oracle.

    The reference is built in fp32 and is itself ``O(N^2 * D)``, so it is the larger of the two
    allocations here; both it and the kernel are inside the out-of-memory skip.
    """
    try:
        t = _inputs(1, N, D, seed=N)
        out = _run(t, eps=1e-5)
        ref = gemm_layernorm_gemm_ref(
            t["a"], t["b"], t["norm_w"], t.get("norm_b"), t["proj_w"], t["proj_b"], eps=1e-5
        )
    except Exception as exc:  # noqa: BLE001 -- re-raised unless it is an OOM
        _skip_if_oom(exc, 1, N, D)
    assert_elementwise(out, ref, _spread_bound(ref, 5e-2), what=f"N={N} D={D}")


@GEMM_LAYERNORM_GEMM.parametrize(
    "N",
    "D",
    only={"N": (200, 384, 640), "D": (136, 192, 320, 512)},
    because=(
        "the rung-isolating test: the einsum and its statistics alone, so a failure in the chain "
        "can be attributed to a stage. The shapes are the ones where the two rungs can disagree "
        "-- an off-grid N (the predicated stats store), a tile-aligned one, and the crossover N -- "
        "crossed with a padded and three unpadded D. The full ladder belongs to test_end_to_end."
    ),
)
def test_einsum_and_stats_rungs(N, D):
    """The token-pair einsum and its per-token statistics, each against its own fp32 reference.

    The two references are deliberately different. The einsum output is compared to the fp32
    product because that is what the kernel computes and then narrows. The statistics are compared
    to the fp32 product's moments -- NOT to the moments of the narrowed output the kernel stored --
    because the kernel accumulates them from the fp32 accumulator, before the narrowing. Using the
    stored tensor as the statistics reference would bake in a rounding the kernel never applied,
    and would loosen the tolerance far enough to hide a real reduction error.
    """
    t = _inputs(1, N, D, seed=N * D)
    tri, mean, rstd = _token_pair_gemm_stats(t["a"], t["b"], eps=1e-5)
    ref = torch.einsum("bdik,bdjk->bdij", t["a"].float(), t["b"].float()).permute(0, 2, 3, 1)
    assert tuple(tri.shape) == (1, N, N, D)
    assert_elementwise(tri, ref, _spread_bound(ref, 2e-2), what=f"N={N} D={D} einsum")

    ref_mean = ref.mean(-1)
    ref_rstd = (ref.var(-1, unbiased=False) + 1e-5).rsqrt()
    assert_elementwise(mean, ref_mean, _spread_bound(ref_mean, 5e-3), what=f"N={N} D={D} mean")
    assert_elementwise(rstd, ref_rstd, _spread_bound(ref_rstd, 1e-2), what=f"N={N} D={D} rstd")


@GEMM_LAYERNORM_GEMM.parametrize_unsupported("arg_fault")
def test_arg_fault_raises(arg_fault, expected_error, expected_match):
    """Every declared caller mistake must be refused at this kernel's OWN front door.

    Each fault is applied to an otherwise well-formed call, so what is under test is the check and
    not some other malformation. The assertion is that the call RAISES -- not that it fails --
    because the hazard these guard against is a plausible wrong answer, which an xfail would treat
    as success.
    """
    N, D = 128, 128
    t = _inputs(1, N, D, has_gate3=True, seed=3)
    kw = dict(eps=1e-5)
    if arg_fault == "N_not_mult8":
        t = _inputs(1, 132, D, has_gate3=True, seed=3)
    elif arg_fault == "D_not_mult8":
        t = _inputs(1, N, 132, has_gate3=True, seed=3)
    elif arg_fault == "bad_direction":
        kw["direction"] = "both"
    elif arg_fault == "a_dtype":
        # BOTH activations and the projection weight in fp32: casting only `a` would be refused by
        # the proj_w-dtype check first, and this region is about the activation dtype itself.
        t = _inputs(1, N, D, dtype=torch.float32, has_gate3=True, seed=3)
    elif arg_fault == "proj_w_dtype":
        t["proj_w"] = t["proj_w"].float()
    elif arg_fault == "norm_w_extent":
        t["norm_w"] = t["norm_w"][: D - 8].contiguous()
    elif arg_fault == "proj_w_extent":
        t["proj_w"] = t["proj_w"][: D - 8].contiguous()
    elif arg_fault == "gate3_extent":
        t["gate3"] = t["gate3"][: N * N - 8].contiguous()
    elif arg_fault == "b_dtype":
        # `a` keeps its supported dtype; only its PARTNER is wrong. Individually both are fine --
        # the fault exists solely in the relationship, which is the class nothing checked.
        t["b"] = t["b"].double()
    else:
        raise AssertionError(f"unhandled arg_fault {arg_fault!r}")
    with front_door_raises(expected_error, expected_match):
        _run(t, **kw)


@matrix_exempt(
    "the subject is the size FORMULA, not a kernel launch: these N values are the two straddling "
    "the bisected crossover plus the swept points either side of it, chosen because the boundary "
    "is what is under test. Drawing them from the shape pool would silently re-target the test "
    "whenever a shape is added for an unrelated reason."
)
@pytest.mark.parametrize(
    "N,expect_continuous",
    [(128, True), (256, True), (512, True), (640, True), (768, False), (1024, False)],
)
def test_heuristic_crossover(N, expect_continuous):
    """The size heuristic picks the continuous schedule at or below the bisected crossover.

    Not drawn from the matrix: the subject is the FORMULA, whose boundary at 640 is the point of
    the test, and the two shapes straddling it are chosen for that reason rather than from the
    kernel's shape pool. The device is left as None so no architecture warning fires and the
    tuned-architecture threshold is the one under test.
    """
    assert _heuristic_cont_pipe(N) is expect_continuous
    assert _CONT_PIPE_BY_ARCH["H200_SXM5"]["CONT_PIPE_MAX_N"] == 640


# ── the three things the shipped module's own gates cannot observe ────────────────────────────
@GEMM_LAYERNORM_GEMM.parametrize(
    "N",
    "D",
    "has_gate3",
    cells=[(128, 128, False), (128, 128, True)],
    because=(
        "the subject is the autotuner's DISPATCH, not a shape: its one knob picks between two "
        "schedules that are bit-identical, so a second shape would re-prove the same equality at "
        "the cost of another sweep. `has_gate3` IS swept, because the tuning key must separate the "
        "two functional variants and it does so by tensor PRESENCE rather than by a declared "
        "scalar -- which is the part that could silently stop holding."
    ),
)
def test_autotune_dispatch_agrees_with_the_fixed_path(N, D, has_gate3):
    """``do_autotune=True`` runs, dispatches, and moves no bit -- and its winner is readable.

    The autotuned path is production-reachable: it is the ``@autotune`` gate on the front door, so
    under CLAUDE.md it must work rather than merely exist. Nothing else in this module reaches it.

    Because the one tunable knob picks between two BIT-IDENTICAL schedules, "the tuned dispatch
    equals the fixed path" is a pure plumbing assertion -- candidate generation, the request key,
    the timing loop and the dispatch all have to be right for it to hold, and none of them can be
    excused by "the other schedule happened to be faster today".

    **The last line is the point of the parametrization.** ``wrapper.autotuner.best_config`` is
    where this tree keeps the winner; ``wrapper.best_config`` -- which is what the upstream this
    was ported from spelled, and the first thing a porter reaches for -- raises ``AttributeError``.
    Reading it here means a future port of this accessor cannot pass silently. The sibling trap is
    quieter and worth naming even though this module does not hit it: ``autotune.freeze(config)``
    returns knobs as keyword arguments for a NON-tuned entry, and the tuned wrapper REFUSES them;
    ``_config=`` is the pin. This chain uses the ``gate=`` idiom and has no freeze helper, so only
    the accessor half applies.

    ``gate3`` is swept because it enters the tuner's request key by being a tensor rather than by
    being named in ``key=[...]``: ``_request_key`` appends one part per ``torch.Tensor`` argument
    and skips ``None``. The upstream keyed ``has_gate3``/``has_bias`` explicitly and this port does
    not, so if that presence behaviour ever changed, the sweep measured for the gated build would
    be reused for the ungated one -- a different compiled kernel.
    """
    t = _inputs(1, N, D, has_gate3=has_gate3, seed=5)
    fixed = _run(t, eps=1e-5)
    tuned = _run(t, eps=1e-5, do_autotune=True)
    assert_bitwise(
        fixed,
        tuned,
        what=f"N={N} D={D} gate3={has_gate3}: the autotuned dispatch differs from the fixed path. Its only knob picks between two bit-identical schedules, so this is a plumbing failure and not a numerical one. max|diff| = {(fixed.float() - tuned.float()).abs().max().item():.3e}",
    )
    best = gemm_layernorm_gemm.autotuner.best_config
    assert best is not None and best.get("cont_pipe") in (True, False), (
        f"after a tuned call the winner must be readable at wrapper.autotuner.best_config; got "
        f"{best!r}"
    )


@matrix_exempt(
    "asserts an in-process CACHE property -- that a repeat call with identical tensor metadata adds "
    "no compile-cache entry -- so sweeping shapes would only re-derive the same key at the cost of "
    "another three compiles per cell"
)
def test_compile_cache_is_reused_across_calls():
    """A second call with the same tensor METADATA adds no compile-cache entry.

    ``cute.compile`` does not cache across calls and this chain compiles THREE kernels, so without
    the module's own ``_COMPILE_CACHE`` every call recompiles all three -- roughly four seconds --
    and a benchmark loop measures the compiler instead of the kernel. That makes the cache
    load-bearing for the perf gate in ``tests/perf/``, whose pinned medians would silently become
    compile times.

    **Nothing else in the suite would notice if it stopped being hit.** Every correctness test here
    would still pass, just slower, which is exactly the shape of defect that survives a green
    suite. The assertion is that the cache does not GROW: a compiled callable is bound to
    shape/stride/dtype plus the tile parameters and not to the storage, so fresh tensors with the
    same metadata must reuse every entry.
    """
    t = _inputs(1, 64, 64, has_gate3=True, seed=2)
    _run(t, eps=1e-5)
    after_first = len(_glg._COMPILE_CACHE)
    assert after_first >= 3, (
        f"expected at least three entries, one per kernel in the chain, got {after_first}. Fewer "
        f"than three means the chain being measured is not this chain."
    )
    fresh = _inputs(1, 64, 64, has_gate3=True, seed=99)
    _run(fresh, eps=1e-5)
    assert len(_glg._COMPILE_CACHE) == after_first, (
        f"the compile cache grew from {after_first} to {len(_glg._COMPILE_CACHE)} on a repeat call "
        f"with identical tensor metadata, so every call is recompiling three kernels (~4 s). Any "
        f"benchmark of this chain is then a measurement of the compiler."
    )


@matrix_exempt(
    "asserts the PURE host-side tile resolver against the feature extents that select each of its "
    "branches; the subject is the arithmetic that picks a tile, which allocates nothing and "
    "launches nothing, so the shape pools do not apply"
)
@pytest.mark.parametrize(
    "D,blk_n,blk_kf",
    [
        (512, 256, 64),  # capped at 256; the contraction tile stays full
        (384, 192, 64),  # largest %16 divisor at most 256
        (256, 256, 64),
        (128, 128, 64),
        (208, 208, 16),  # %16 but not %64 -> the two tiles move INDEPENDENTLY
        (144, 144, 16),
        (96, 96, 32),
        (48, 48, 16),
        (24, 16, 16),  # 8 mod 16 -> no %16 divisor exists, so the output tile falls to 16
        (72, 16, 16),
        (264, 16, 16),
        (8, 16, 16),  # the minimum D: the output tile is WIDER than the feature axis
    ],
)
def test_tile_resolution_covers_every_feature_extent(D, blk_n, blk_kf):
    """The resolver picks the documented block for each class of ``D``, including the awkward ones.

    **This arithmetic is what lets ``D`` need only ``D % 8 == 0``**, and it is one edit from
    silently narrowing that. Getting it wrong does not raise -- it writes the wrong number of
    columns -- so nothing else here would catch a regression in it.

    Three rows carry the whole design and none is decoration:

    * ``D=208`` is a multiple of 16 but not of 64, so the OUTPUT tile stays at the full 208 while
      the CONTRACTION tile must shrink to 16. The two are resolved independently, and a rule that
      tied them would pass every other row in this list and fail only here.
    * ``D=24`` has no multiple-of-16 divisor at all, so the output tile falls to 16 and the write
      becomes predicated. A resolver that returned 24 would build a tile the swizzle atom cannot
      express.
    * ``D=8`` makes the output tile WIDER than the feature axis, which is the case the output
      over-allocation exists for: without it the last tile's TMA address leaves the allocation.
    """
    plan = GemmLayerNormGemmSm90(torch.bfloat16, 1, 128, D)
    assert (plan.blk_n, plan.blk_kf) == (blk_n, blk_kf), (
        f"D={D}: resolved (blk_n, blk_kf) = ({plan.blk_n}, {plan.blk_kf}), expected "
        f"({blk_n}, {blk_kf})"
    )
    assert plan.blk_n % 16 == 0, (
        f"D={D}: the output tile must stay a multiple of the shared-memory swizzle atom, got "
        f"{plan.blk_n}"
    )


# ── the LayerNorm's own bias is optional, and the two builds must not share a kernel ───────────
@GEMM_LAYERNORM_GEMM.parametrize(
    "N",
    "D",
    "has_norm_bias",
    "has_gate3",
    cells=[
        (128, 128, False, False),
        (128, 128, True, False),
        (128, 128, False, True),
        (136, 72, False, False),  # off-grid N with the PADDED projection path
        (200, 208, False, False),  # off-grid N, %16-not-%64 D
    ],
    because=(
        "the axis under test is the presence of op 8's shift, which is a compile-time branch in the "
        "normalize prologue rather than a shape behaviour -- so the shapes are three: a tile-aligned "
        "control, and the two that turn the projection's two predication paths on, since the "
        "zero-staging shares its scratch with the contraction-tile padding and a bug would surface "
        "where both are active. `has_gate3` is swept at one shape only to show the flags compose."
    ),
)
def test_norm_bias_optional_matches_reference(N, D, has_norm_bias, has_gate3):
    """A LayerNorm with no bias computes the unbiased normalize, at every predication path.

    ``norm_b=None`` is not a runtime branch: it compiles a build that stages ``0.0`` into the affine
    scratch, which is the value the contraction-tile padding already writes there, so the
    normalize's ``+ norm_b[k]`` becomes ``+ 0.0``. That makes the bias-free result exactly the
    unbiased LayerNorm rather than an approximation of one, and the reference omits the shift the
    same way by passing None straight to ``F.layer_norm``.

    The off-grid cells matter more than the aligned one here: the zero-staging writes into the SAME
    scratch as the ``D``-padding gate, so a mistake that put the two in the wrong order would show
    up only where both are active.
    """
    t = _inputs(1, N, D, has_norm_bias=has_norm_bias, has_gate3=has_gate3, seed=N + D)
    assert ("norm_b" in t) is has_norm_bias
    out = _run(t, eps=1e-5)
    ref = gemm_layernorm_gemm_ref(
        t["a"],
        t["b"],
        t["norm_w"],
        t.get("norm_b"),
        t["proj_w"],
        t.get("proj_b"),
        eps=1e-5,
        gate3=t.get("gate3"),
    )
    assert_elementwise(
        out,
        ref,
        _spread_bound(ref, 5e-2),
        what=f"N={N} D={D} norm_bias={has_norm_bias}",
    )


@matrix_exempt(
    "the subject is the COMPILE-CACHE KEY, not a shape: it asserts that two builds which differ "
    "only by a flag do not share one cached artifact. One shape is enough and a second would only "
    "re-derive the same key"
)
@pytest.mark.parametrize("first", [True, False], ids=["biased_first", "unbiased_first"])
def test_norm_bias_variants_do_not_share_a_compiled_kernel(first):
    """Both ``norm_b`` builds in ONE process, in BOTH orders, each still computing its own answer.

    **This is the test that would fail if ``has_norm_bias`` were dropped from the
    ``_COMPILE_CACHE`` key**, and it is here because that omission is not hypothetical: the same
    class of key defect -- a cached artifact selected on a key that does not separate two builds --
    was the leading explanation of a live illegal memory access in this chain's front half at the
    time this flag was added. Whichever variant compiled FIRST would be handed to the second.

    **MEASURED failure mode, rather than the assumed one.** Simulating the omission -- a cache
    wrapper that strips the last key field -- the second call raises
    ``TypeError: Mismatched type on argument #4`` from the tvm-ffi boundary, in BOTH orders. It does
    not return a wrong answer and it does not fault: the compiled entry's signature records whether
    that operand is a tensor or None, and the FFI refuses the mismatch. So this test fails loudly
    without the key change, which is what it is for, but the hazard it guards is a refused launch
    rather than silent corruption. Worth stating precisely, because the same reasoning applied to
    the front half predicts a TypeError there too -- and that fault is an out-of-bounds READ, so a
    plain presence/absence key collision is probably NOT its mechanism.

    Both orders are swept anyway: the FFI check is what makes this loud today, and it is a property
    of the launch path rather than of the cache. A future change that made the operand
    unconditionally present -- a zero tensor instead of None, say -- would remove the type
    difference and with it the diagnostic, leaving a silent wrong answer in exactly one order.

    The assertion is deliberately in two parts. That the two outputs DIFFER rules out a shared
    kernel -- if the cache collided they would be bit-identical, which is exactly the symptom to
    catch. That each matches its OWN reference rules out the opposite failure, two distinct kernels
    both computing something wrong.
    """
    N, D = 128, 128
    order = [first, not first]
    got, refs = {}, {}
    for has_nb in order:
        t = _inputs(1, N, D, has_norm_bias=has_nb, seed=17)
        got[has_nb] = _run(t, eps=1e-5)
        refs[has_nb] = gemm_layernorm_gemm_ref(
            t["a"], t["b"], t["norm_w"], t.get("norm_b"), t["proj_w"], t.get("proj_b"), eps=1e-5
        )
    # Two builds, two cache entries -- the key must separate them.
    feature_keys = [k for k in _glg._COMPILE_CACHE if k[0] == "featuregemm"]
    variants = {k[-1] for k in feature_keys}
    assert variants == {True, False}, (
        f"expected the compile cache to hold a separate 'featuregemm' entry per norm-bias variant; "
        f"the last key field took values {variants} over {len(feature_keys)} entries. If it is a "
        f"single value, `has_norm_bias` is missing from the key and one build is being reused for "
        f"the other."
    )
    assert not torch.equal(got[True], got[False]), (
        "the biased and unbiased builds produced BIT-IDENTICAL output, which they cannot: the "
        "reference differs between them. That is the signature of one compiled kernel serving both."
    )
    for has_nb in (True, False):
        assert_elementwise(
            got[has_nb],
            refs[has_nb],
            _spread_bound(refs[has_nb], 5e-2),
            what=(
                f"norm_bias={has_nb} (compiled {'first' if has_nb == first else 'second'}) against "
                f"its OWN reference -- exceeding this is the signature of the variant having run "
                f"the other one's kernel"
            ),
        )
