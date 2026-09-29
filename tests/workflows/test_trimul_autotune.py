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
"""Tests for `fold_cp_ops.workflows.trimul_autotune` -- the cp=1 TriMul dispatcher.

**The highest-value test here needs no GPU and is not a numerical one.** The defect family this
module documents is a size heuristic that returns a combo the shape cannot run, and it recurs one
branch at a time: the kernel this ports from fixed two instances and shipped a third and a fourth.
`test_heuristic_pick_always_survives_prune` closes it by BRUTE FORCE -- every combo the heuristic
returns, over thousands of shapes, must survive the pruner for that same shape. A branch that
forgets a floor fails it immediately, whichever branch it is.

**Two gates, chosen per property.** `_gemm1` is ONE bf16 GEMM, so it is checked against a torch
``einsum`` under a DERIVED bound. The end-to-end chain is five fused stages whose derived bound would
be so loose as to assert nothing, so it is checked against an fp32 oracle under an EMPIRICAL one,
with the measurement that set it written down beside it. Saying which is which is the point: one of
them would catch a one-ulp regression and the other would not.

**Both are ELEMENT-WISE.** Every comparison here judges each element against its own allowance and
reports the worst INDEX with both values, never a pooled scalar. A pooled figure -- and these tests
did once assert ``max|diff| / max|ref|`` -- divides the worst deviation by the LARGEST reference
magnitude, so an element that is small but wrong by 100% is invisible. That is the shape of every
defect this repo keeps finding: a tile edge, a padded lane, a straddle row, a last-wave CTA, a
handful of elements out of hundreds of millions. See `_e2e_bound` for why the chain's allowance
cannot be relative-only.
"""

import itertools

import pytest
import torch

import fold_cp_ops.workflows.trimul_autotune as T
from fold_cp_ops.testing.kernel_matrix import Axis, KernelMatrix, Unsupported, matrix_exempt
from fold_cp_ops.testing.numeric_guard import numeric_exempt
from fold_cp_ops.testing.numerics import assert_bitwise, assert_elementwise, gemm_error_bound

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(_SM != 9, reason="fold-cp-ops kernels require SM90 (H100/H200)")


#: The per-element allowance for the chain, as ``atol + rtol * |ref_i|``.
#:
#: **Why not a pooled scalar.** These tests used to assert ``max|diff| / max|ref| <= 0.02``,
#: which divides the worst deviation by the LARGEST reference magnitude. An element whose own value
#: is small but wrong by 100% contributes a ``|diff|`` negligible against ``max|ref|`` and is
#: invisible -- and this is the e2e workflow, the largest tensor in the repo, so it has the most room
#: to hide a local defect. Judged per element instead, with the worst index and both values reported.
#:
#: **Why not a pure relative bound.** MEASURED, over (128,128), (64,136) and (64,256): stratified by
#: reference magnitude, the max relative error is 0.019-0.022 for elements above 10% of ``max|ref|``,
#: 0.15-0.19 between 1% and 10%, and **1126 to 6227** below 1%. Near-zero outputs come from
#: cancellation and their relative error is unbounded, so a relative-only bound cannot be written.
#: Absolute error, by contrast, is tightly held: ``max|diff|`` was 0.025-0.031 against a ``max|ref|``
#: of 4.5-5.7, i.e. a pooled 0.0053-0.0061.
#:
#: So the floor is a FRACTION OF THE REFERENCE'S SCALE rather than a constant, which is what lets it
#: transfer across cells whose output magnitude differs. At the measured cells this is ~1.6x the
#: worst observed deviation for a near-zero element and ~3x for a large one -- and it is STRICTLY
#: TIGHTER than the pooled bar it replaces (0.051 vs 0.103 for a small element at those cells).
_E2E_ABS_FRAC = 0.01
_E2E_REL_PER_ELEM = 0.01


def _e2e_bound(ref):
    """Per-element allowance for the five-stage chain, shaped like `ref`.

    Args:
        ref: The fp32 oracle output. Its MAXIMUM magnitude sets the absolute floor, so the floor
            scales with the cell instead of being a constant that only holds at one output scale.

    Returns:
        A tensor broadcastable to `ref`: ``_E2E_ABS_FRAC * max|ref| + _E2E_REL_PER_ELEM * |ref|``.
    """
    return _E2E_ABS_FRAC * ref.abs().max() + _E2E_REL_PER_ELEM * ref.abs()


TRIMUL_AUTOTUNE = KernelMatrix(
    kernel="trimul_autotune",
    axes=(
        Axis(
            name="N",
            domain=(
                "the token extent. Any positive int meeting the 16-byte floor: the token-pair "
                "einsum contracts (N, N) operands, so N is their row pitch and N % 8 == 0 is "
                "required for a 16-bit activation. NOT restricted to a power of two or a tile "
                "multiple"
            ),
            values=(8, 40, 64, 99, 100, 128, 136, 192, 256, 512, 2048, 4096),
            facets={
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                "off_grid": lambda v: v % 128 != 0,
                "unaligned": lambda v: v % 8 != 0,
                "large": lambda v: v >= 256,
                # The A2A-fused workflow's OWN token counts. This axis IS ``N_token``, so the
                # declared ladder applies to it directly and not through the ``M = N_token**2 / cp``
                # relation the projection kernels use -- the dispatcher is handed the un-sharded
                # token extent. 8192 and 12288 are deliberately absent: at D=128 the activation
                # alone is ``N**2 * D * 2`` bytes, i.e. 17 GB and 39 GB, before any intermediate,
                # so they belong to the perf cells. See
                # `test_the_dispatcher_runs_at_the_workflow_token_counts` for what the two present
                # values do and do not check.
                "workflow_N_token": lambda v: v in (2048, 4096),
            },
        ),
        Axis(
            name="D",
            domain=(
                "the feature width. 16-byte aligned; NOT restricted to a multiple of 32, which is "
                "the whole reason the dispatcher keeps a universal fallback"
            ),
            values=(32, 64, 128, 136, 192, 200, 256, 384, 512, 132),
            facets={
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                "mod32": lambda v: v % 32 == 0,
                "off_grid": lambda v: v % 32 != 0,
                "unaligned": lambda v: v % 8 != 0,
                # The four TriMul feature widths, under the table's own name. Alongside the
                # generic facets, which describe alignment regimes rather than the widths
                # the workflow runs.
                "workflow_D": lambda v: v in (128, 256, 384, 512),
            },
        ),
        Axis(
            name="B",
            domain="the batch; it only enlarges the flattened token axis M = B * N * N",
            values=(1, 2, 3),
            facets={"single": lambda v: v == 1, "batched": lambda v: v > 1},
        ),
        Axis(
            name="direction",
            domain="which axis the token-pair einsum contracts",
            values=("outgoing", "incoming", "sideways"),
            facets={
                "outgoing": lambda v: v == "outgoing",
                "incoming": lambda v: v == "incoming",
                "invalid": lambda v: v not in ("outgoing", "incoming"),
            },
        ),
        Axis(
            name="has_mask",
            domain="whether a per-token mask is supplied; it masks the DUAL only, never the gate",
            values=(False, True),
            facets={"masked": lambda v: v, "unmasked": lambda v: not v},
        ),
        Axis(
            name="has_bias",
            domain="whether the six optional bias vectors are supplied",
            values=(False, True),
            facets={"biased": lambda v: v, "unbiased": lambda v: not v},
        ),
        Axis(
            name="variant",
            domain="the combo, a member of trimul_autotune.TRIMUL_VARIANTS",
            values=T.TRIMUL_VARIANTS,
            facets={
                # The family, which used to be the phase-label PREFIX and is now a declared
                # tuple. A `startswith` test here would silently classify every variant as
                # back-gate once the names stopped carrying the prefix -- the facet would still
                # evaluate, still return a bool, and quietly cover nothing.
                "frontgate": lambda v: v in T.FRONTGATE_VARIANTS,
                "backgate": lambda v: v in T.BACKGATE_VARIANTS,
                "runnable_here": lambda v: (
                    v in ("gemm__layernorm__gemm_hadamard", "gemm__layernorm_gemm")
                ),
            },
        ),
        Axis(
            name="front_v",
            domain="the front fusion for a FRONT-GATE combo, a member of trimul_autotune.FRONT_VARIANTS",
            values=T.FRONT_VARIANTS,
            facets={
                "fused": lambda v: v != "lng2k",
                "separate_ln": lambda v: v == "lng2k",
                "alg_fold": lambda v: v == "alg_fold",
            },
        ),
        Axis(
            name="row_mean",
            domain=(
                "any float, added to every element of `x`. NOT a dispatcher parameter -- it is a "
                "property of the INPUT, and it earns an axis because the chain opens with a "
                "LayerNorm over `x` itself, so the caller sets the conditioning of every row it "
                "normalizes. MEASURED before the axis existed: worst |err|/bound ratio 0.337 at "
                "mu=0, 0.339 at 1, 0.332 at 10, 0.340 at 100 -- flat, which is the answer a "
                "correct chain gives and exactly what a defect proportional to mean**2 does not"
            ),
            values=(0.0, 10.0, 100.0),
            facets={
                "zero_mean": lambda v: v == 0.0,
                "off_centre": lambda v: v != 0.0,
                "badly_conditioned": lambda v: abs(v) >= 100.0,
            },
            waived={
                "badly_conditioned": (
                    "mu >= 1000 is excluded on purpose: `x` is bf16, whose spacing at 1000 is ~3.9, "
                    "so a unit-variance row quantizes to a handful of levels and its variance "
                    "becomes quantization noise. Measured worst ratio 0.930 there -- still inside "
                    "the bound, but it measures the INPUT's conditioning, not the chain's. 100 is "
                    "the largest offset at which the probe is still about the kernel"
                )
            },
        ),
    ),
    # The whole cp=1 chain: LayerNorm, the gated projections, the token-pair einsum and
    # its statistics, and the output projection.
    computes=("row_reduction", "contraction", "saturating_activation"),
    unsupported=(
        Unsupported(
            where=lambda N, D, direction: N % 8 != 0 and D == 128 and direction == "outgoing",
            raises=ValueError,
            match=r"N must be 16-byte aligned",
            reason=(
                "the token-pair einsum contracts (N, N) operands, so N is their row pitch and an "
                "unaligned one cannot be addressed by a TMA descriptor. NO combo can run it, which "
                "is why the front door refuses the shape rather than returning a combo that will "
                "then refuse it -- the whole defect family this module exists to close"
            ),
        ),
        Unsupported(
            where=lambda N, D, direction: D % 8 != 0 and N == 128 and direction == "outgoing",
            raises=ValueError,
            match=r"D must be 16-byte aligned",
            reason=(
                "the same 16-byte floor on the feature width, which is the contraction extent of "
                "the front and of both output projections"
            ),
        ),
        Unsupported(
            where=lambda N, D, direction: (
                direction not in ("outgoing", "incoming") and N == 128 and D == 128
            ),
            raises=ValueError,
            match=r"direction must be",
            reason=(
                "the direction selects which axis the einsum contracts, and there are exactly two. "
                "An unrecognised one would otherwise fall through to the outgoing branch and "
                "return a plausible tensor for a different operation"
            ),
        ),
    ),
    arg_faults_waived=(
        "this module is a DISPATCHER over kernels that each carry their own argument-fault axis; "
        "its own arguments are the workflow's weights, whose shapes it checks and whose dtypes it "
        "casts. The faults worth naming here are shape ones, and they are the three regions above"
    ),
)


def _shape_stub(B, N, D):
    """A ``(B, N, N, D)``-shaped CUDA tensor holding ONE element of storage.

    Purpose
        `_prune` and the heuristic read a shape and a device and nothing else, so the brute-force
        test would otherwise allocate terabytes to ask a question about integers.

    Args:
        B: Batch.
        N: Token extent.
        D: Feature width.

    Returns:
        A zero-strided view. **Never usable as a kernel input** -- every element aliases the same
        address, so anything that reads it reads one value.
    """
    return torch.empty(1, device="cuda", dtype=torch.bfloat16).as_strided((B, N, N, D), (0,) * 4)


def _weights(B, N, D, dtype=torch.bfloat16, bias=True, mask=False, row_mean=0.0, seed=0):
    """Build one well-formed argument set for `trimul_autotuned`.

    Purpose
        One builder for every numerical test, so a failing cell cannot be a cell that was built
        differently. **Correctness-only, NOT performance-representative.**

    Semantics
        The projections are scaled by ``1/sqrt(D)`` so the pre-activations do not grow with the
        feature width, which would otherwise make a wide cell look systematically worse than a
        narrow one for a reason that is not the dispatcher's.

    Args:
        B, N, D: The shape.
        dtype: The activation and weight dtype.
        bias: Whether to build all six optional bias vectors.
        mask: Whether to build a per-token mask.
        row_mean: A constant added to every element of ``x`` AFTER it is drawn, so every row the
            chain's input LayerNorm normalizes is offset together. In exact arithmetic the mean is
            subtracted straight back out; in floating point it is a cancellation whose conditioning
            grows with ``|mu|/sigma``, and ``torch.randn`` alone never leaves ``mu = 0``.
        seed: Generator seed, so a failing cell is reproducible from its parameters.

    Returns:
        A kwargs dict for `trimul_autotuned`, minus ``direction``, ``eps`` and the selection knobs.
    """
    g = torch.Generator(device="cuda")
    g.manual_seed(seed if seed else B * 1000 + N * 10 + D)
    r = lambda *s, d=dtype: torch.randn(*s, device="cuda", dtype=d, generator=g)  # noqa: E731
    kw = dict(
        x=r(B, N, N, D),
        norm_in_w=r(D, d=torch.float32),
        norm_out_w=r(D, d=torch.float32),
        p_in_w=r(2 * D, D) / D**0.5,
        g_in_w=r(2 * D, D) / D**0.5,
        p_out_w=r(D, D) / D**0.5,
        g_out_w=r(D, D) / D**0.5,
    )
    if row_mean:
        kw["x"] = (kw["x"].float() + row_mean).to(dtype)
    if bias:
        kw.update(
            norm_in_b=r(D, d=torch.float32),
            norm_out_b=r(D, d=torch.float32),
            p_in_b=r(2 * D, d=torch.float32),
            g_in_b=r(2 * D, d=torch.float32),
            p_out_b=r(D, d=torch.float32),
            g_out_b=r(D, d=torch.float32),
        )
    if mask:
        kw["mask"] = (torch.rand(B, N, N, device="cuda", generator=g) > 0.3).to(dtype)
    return kw


def _oracle(kw, direction, eps=1e-5):
    """The fp32 reference for one argument set.

    Args:
        kw: A `_weights` dict.
        direction: ``"outgoing"`` or ``"incoming"``.
        eps: The variance floor; must match what the kernel was given.

    Returns:
        ``(B, N, N, D)`` fp32.
    """
    return T.trimul_ref(
        kw["x"],
        direction,
        kw.get("mask"),
        kw["norm_in_w"],
        kw.get("norm_in_b"),
        kw["p_in_w"],
        kw["g_in_w"],
        kw["norm_out_w"],
        kw.get("norm_out_b"),
        kw["p_out_w"],
        kw["g_out_w"],
        kw.get("p_in_b"),
        kw.get("g_in_b"),
        kw.get("p_out_b"),
        kw.get("g_out_b"),
        eps=eps,
    )


# ── the invariant: what closes the defect family ───────────────────────────────────────────────


@requires_sm90
@TRIMUL_AUTOTUNE.parametrize("B")
@numeric_exempt(
    "asserts every heuristic pick survives _prune; a pure host-side validity sweep, no kernel"
)
def test_heuristic_pick_always_survives_prune(B):
    """EVERY combo the size heuristic returns must survive the pruner for that SAME shape.

    Purpose
        The highest-value test in this file, and the one the whole module is shaped around. The
        defect it closes has recurred at least four times upstream: a branch of the regime tree
        returns a combo whose kernels cannot run the shape, the pruner would have rejected it, and
        because the heuristic is the SHIPPED default nothing else stands between the caller and a
        kernel-level refusal.

    Semantics
        Brute force over the whole grid rather than a sample, because the failures are sparse and
        structural -- a missing floor bites at exactly the widths the sampler skips. The grid spans
        every feature width from 8 to 1056 in steps of 8 crossed with a token ladder that includes
        the unaligned cases, both directions and both mask states.

        Shapes NO combo can run are skipped rather than asserted on: at an unaligned token extent
        the front door raises before selection, which the refusal tests cover. What is asserted for
        every RUNNABLE shape is both directions of the property -- the heuristic's pick survives,
        AND the pool is non-empty, so "survives" cannot be satisfied vacuously by a pruner that
        keeps everything or by a heuristic that returns something the pool never contained.

    Args:
        B: The batch, from the matrix. Split across parameters rather than looped inside so a
            failure names which batch it was.
    """
    pool = [T._as_autotune_config(c) for c in T._trimul_configs()]
    checked = 0
    for N, D in itertools.product(
        (8, 16, 40, 64, 96, 99, 100, 128, 136, 192, 200, 201, 256, 320, 384, 512, 1024, 2048),
        tuple(range(8, 1064, 8)),
    ):
        if not (T._gemm1_validity(N) and T._front_validity(D)):
            continue  # refused at the front door before any combo is chosen
        x = _shape_stub(B, N, D)
        for nob in (torch.empty(1, device="cuda"), None):
            # The BACK LayerNorm's bias is swept to assert it does NOT affect prunability. It used
            # to: `gemm_layernorm_gemm` declared that argument non-optional, so a None reached it
            # as an AttributeError and `_back_validity` gated `glg` off. That was a defect in the
            # kernel (identical in `main`), fixed at the source, and the gate came out with it.
            # Keeping the sweep is what would catch someone reintroducing a validity gate on an
            # argument that is no longer a validity input.
            args = {"x": x, "norm_out_b": nob}
            assert T._prune(pool, args), (
                f"B={B} N={N} D={D} norm_out_b={'given' if nob is not None else 'None'} is "
                f"runnable but the pool is EMPTY -- the pruner and the front door's own floors "
                f"disagree about what this shape can do"
            )
            for direction, has_mask in (("outgoing", False), ("incoming", True)):
                cfg = T._trimul_heuristic_config(N, D, direction, has_mask, B=B)
                assert T._prune([T._as_autotune_config(cfg)], args), (
                    f"the size heuristic returned {cfg} for B={B} N={N} D={D} "
                    f"direction={direction} has_mask={has_mask} "
                    f"norm_out_b={'given' if nob is not None else 'None'}, and `_prune` REJECTS "
                    f"it. That combo is what the shipped select='heuristic' default would run, so "
                    f"this is a kernel-level refusal reachable from the default path."
                )
                checked += 1
    assert checked > 6000, f"the grid collapsed to {checked} picks; it is meant to be exhaustive"


@matrix_exempt("the subject is the GRID's own contents, which no shape axis parametrizes")
def test_grid_offers_only_variants_this_tree_implements():
    """The candidate pool never contains a config nothing can build.

    A grid that can hand back an unimplemented combo is a crash reachable from
    ``select="autotune"``, which is why the pool is narrowed WITH the kernel rather than left wide
    and pruned later. `layernorm_gemm` builds one LayerNorm fusion, so ``gemm__layernorm_gemm`` contributes three
    candidates and not six.
    """
    configs = T._trimul_configs()
    assert len(configs) == 11, [str(c) for c in configs]
    assert {c.variant for c in configs} == {
        "gemm_layernorm_gemm",
        "gemm__layernorm__gemm_hadamard",
        "gemm__layernorm_gemm",
        "dual_gated_gemm__gemm__layernorm_dual_gated_gemm",
    }
    assert sum(c.variant == "gemm__layernorm_gemm" for c in configs) == 3
    for c in configs:
        assert c.variant in T.TRIMUL_VARIANTS
        if c.variant == "gemm__layernorm_gemm":
            assert c.lng_inner in T.LNG_INNER_VARIANTS
        if c.variant in T.FRONTGATE_VARIANTS:
            assert c.front_v in T.FRONT_VARIANTS
        else:
            assert c.back_v in T.BACK_VARIANTS


@matrix_exempt("the subject is the config PROJECTION, which is pure host bookkeeping")
def test_projection_keeps_irrelevant_knobs_out_of_the_cache_key():
    """A knob a combo does not consume must not reach its cache key.

    Two candidates that run identical code must share one cache entry. Letting an unread knob into
    the key splits it in two, which does not give a wrong answer -- it doubles a sweep and halves a
    cache hit rate, silently.
    """
    glg = T.TriMulConfig(
        "gemm_layernorm_gemm", front_v="lng2k", lng_inner="prolog_ln", back_v="prolog_ln"
    )
    assert set(glg.all_kwargs()) == {"variant", "front_v"}
    assert T.TriMulConfig("gemm_layernorm_gemm", front_v="lng2k").all_kwargs() == glg.all_kwargs()
    lng = T.TriMulConfig(
        "gemm__layernorm_gemm", front_v="alg_fold", lng_inner="alg_fold", back_v="prolog_ln"
    )
    assert set(lng.all_kwargs()) == {"variant", "front_v", "lng_inner"}
    p2 = T.TriMulConfig(
        "dual_gated_gemm__gemm__layernorm_dual_gated_gemm", front_v="lng2k", back_v="alg_fold"
    )
    assert set(p2.all_kwargs()) == {"variant", "back_v"}
    # The round trip must restore what the combo READS; a projected-away field comes back at its
    # default, which is correct precisely because nothing reads it.
    for c in (glg, lng, p2):
        back = T._from_autotune_config(T._as_autotune_config(c))
        assert back.all_kwargs() == c.all_kwargs(), (c, back)
    assert "an unknown variant keeps every field" and set(
        T.TriMulConfig("P9.nope").all_kwargs()
    ) == {"variant", "front_v", "lng_inner", "back_v"}


# ── the contraction convention ─────────────────────────────────────────────────────────────────


@requires_sm90
@TRIMUL_AUTOTUNE.parametrize(
    "direction",
    "N",
    "B",
    only={"direction": ["outgoing", "incoming"], "N": [40, 64, 128, 136], "B": [1, 2]},
    because=(
        "the SUBJECT is the contraction convention, which depends on the direction and on nothing "
        "else -- the feature width is a batch axis to this GEMM. The token extents span a tile "
        "multiple, an off-grid value and one below a tile; the unaligned values are refused before "
        "this point and are covered by the refusal test"
    ),
)
def test_gemm1_matches_the_einsum(direction, N, B):
    """Op 7 computes the einsum the reference states, in BOTH directions.

    This package's `gemm` computes ``A @ B^T`` while the kernel this dispatcher ports from computes
    ``A @ B``, so the upstream's transposes do NOT carry across: the outgoing case needs none at all
    and the incoming case needs both operands moved. Transliterating the upstream would produce a
    plausible, correctly-shaped, wrong tensor with no error anywhere, which is why this is asserted
    against the einsum rather than against the upstream's call.

    The bound is DERIVED -- one bf16 GEMM over ``K = N`` terms -- because at this level it can be.

    Args:
        direction: Which axis the einsum contracts.
        N: The token extent.
        B: The batch.
    """
    D, M = 32, B * N * N
    g = torch.Generator(device="cuda")
    g.manual_seed(N * 100 + B)
    a = torch.randn(D, M, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(D, M, device="cuda", dtype=torch.bfloat16, generator=g)
    got = T._gemm1(a, b, B, N, D, direction)
    a4 = a.reshape(D, B, N, N).permute(1, 2, 3, 0).float()
    b4 = b.reshape(D, B, N, N).permute(1, 2, 3, 0).float()
    eq = "bikd,bjkd->bijd" if direction == "outgoing" else "bkid,bkjd->bijd"
    ref = torch.einsum(eq, a4, b4)
    got4 = got.reshape(D, B, N, N).permute(1, 2, 3, 0).float()
    # One (N, N) x (N, N) bf16 GEMM per (batch, feature); the bound is that GEMM's, broadcast over
    # the pairs, which is exactly what op 7 is.
    lhs = a.reshape(D, B, N, N)[0, 0]
    bound = gemm_error_bound(lhs, lhs, torch.bfloat16).max()
    assert_elementwise(got4, ref, bound.expand_as(ref), what=f"gemm1 {direction}")


# ── the chains, against the fp32 oracle ────────────────────────────────────────────────────────


@requires_sm90
@TRIMUL_AUTOTUNE.parametrize(
    "variant",
    "front_v",
    "direction",
    only={
        "variant": [
            "gemm_layernorm_gemm",
            "gemm__layernorm__gemm_hadamard",
            "gemm__layernorm_gemm",
            "dual_gated_gemm__gemm__layernorm_dual_gated_gemm",
        ],
        "direction": ["outgoing", "incoming"],
    },
    because=(
        "every combo the swept grid can emit, over both directions. `front_v` is projected away "
        "for dual_gated_gemm__gemm__layernorm_dual_gated_gemm, so it is reused there to sweep the BACK fusion instead -- otherwise those "
        "cells would run one chain three times and `back_v` would go untested. `dual_gated_gemm__gemm__layernorm__gemm_hadamard` is "
        "absent because it is not in the grid and is not expressible here at all"
    ),
)
def test_matches_the_reference(variant, front_v, direction):
    """Each runnable combo reproduces the fp32 oracle, at every front and in both directions.

    The allowance is `_e2e_bound`, per element, empirical, and documented where it is defined.
    What this catches is a STRUCTURAL error -- a wrong contraction, a transposed weight, a bias
    dropped or added at the wrong point in the chain, a mask applied to the output gate instead of
    the dual, a permuted token order. Every one of those moves the answer by O(1).

    Args:
        variant: The combo.
        front_v: The front fusion.
        direction: Which axis op 7 contracts.
    """
    B, N, D = 1, 128, 128
    kw = _weights(B, N, D, bias=True, mask=False)
    back_v = "prolog_ln" if front_v == "prolog_ln" else "alg_fold"
    cfg = T.TriMulConfig(variant, front_v=front_v, lng_inner="alg_fold", back_v=back_v)
    assert T._prune([T._as_autotune_config(cfg)], {"x": kw["x"], "norm_out_b": kw["norm_out_b"]}), (
        f"{cfg} pruned at ({N}, {D})"
    )
    got = T.trimul_autotuned(
        **kw, direction=direction, eps=1e-5, _config=T._as_autotune_config(cfg)
    )
    ref = _oracle(kw, direction)
    assert got.shape == ref.shape and got.dtype == kw["x"].dtype
    assert_elementwise(got.float(), ref, _e2e_bound(ref), what=f"{cfg} {direction}")


@requires_sm90
@TRIMUL_AUTOTUNE.parametrize(
    "row_mean",
    "direction",
    only={"direction": ["outgoing", "incoming"]},
    because=(
        "'sideways' is the value the front door must REJECT -- it computes nothing to "
        "compare, and its refusal is asserted by test_unsupported_raises"
    ),
)
def test_off_centre_rows_do_not_degrade_the_chain(row_mean, direction):
    """Offsetting every row's mean must not move the chain's agreement with the fp32 oracle.

    The chain opens with a LayerNorm over ``x`` itself, so the row mean is an INPUT property the
    caller sets -- and it is the one property no shape axis can vary. What it would expose is a
    padded tile whose masked lanes contribute something rather than nothing: that error is
    proportional to ``mean**2``, hence exactly zero at the mean ``torch.randn`` produces, which is
    how the same defect once survived 598 tests in the standalone LayerNorm.

    MEASURED before this test existed: worst |err|/bound ratio 0.337 / 0.339 / 0.332 / 0.340 at
    mu = 0 / 1 / 10 / 100. Flat, which is why the chain gets an axis rather than a waiver -- and
    why the axis is worth having: a waiver would have recorded the same conclusion with nothing
    left running to notice when it stopped being true.

    Args:
        row_mean: The constant added to every element of ``x``.
        direction: Which axis op 7 contracts. Crossed in because the two directions feed the einsum
            different operand layouts, and only one of them is the contiguous case.
    """
    kw = _weights(1, 128, 128, bias=True, mask=False, row_mean=row_mean)
    got = T.trimul_autotuned(**kw, direction=direction, eps=1e-5, select="heuristic")
    ref = _oracle(kw, direction)
    assert_elementwise(got.float(), ref, _e2e_bound(ref), what=f"row_mean={row_mean:g} {direction}")


@requires_sm90
@TRIMUL_AUTOTUNE.parametrize(
    "has_mask",
    "has_bias",
    "D",
    only={"D": [64, 136, 256]},
    because=(
        "the SUBJECT is the two OPTIONAL inputs, which are independent of the combo -- one width "
        "per k-tiling regime is enough to show they are not entangled with it. 136 additionally "
        "prunes the transposing back, so this is also where the fallback combo is exercised"
    ),
)
def test_mask_and_bias_reach_the_right_operands(has_mask, has_bias, D):
    """The mask reaches the DUAL only and every bias lands at its own point in the chain.

    Op 5 masks ``ab`` before the chunk and leaves the output gate alone. A mask applied to the gate
    instead, or to both, stays finite and plausibly scaled -- so it is caught only by an oracle that
    composes the ops explicitly, which is what `trimul_ref` does.

    Args:
        has_mask: Whether a mask is supplied.
        has_bias: Whether the six bias vectors are supplied.
        D: The feature width.
    """
    B, N = 1, 64
    kw = _weights(B, N, D, bias=has_bias, mask=has_mask)
    got = T.trimul_autotuned(**kw, direction="outgoing", eps=1e-5, select="heuristic")
    ref = _oracle(kw, "outgoing")
    assert_elementwise(
        got.float(), ref, _e2e_bound(ref), what=f"D={D} mask={has_mask} bias={has_bias}"
    )


@requires_sm90
@TRIMUL_AUTOTUNE.parametrize(
    "B",
    "N",
    only={"N": [64, 136]},
    because=(
        "the SUBJECT is the batch, which only enlarges the flattened token axis -- one aligned and "
        "one off-grid token extent is enough to show the (b, i, j) order survives the flattening "
        "and the op-7 batching over L = d*B + b"
    ),
)
def test_batch_preserves_the_token_order(B, N):
    """A batched call gives every batch the answer it would get alone.

    The token axis is flattened to ``M = B * N * N`` and op 7 fuses the feature and batch axes into
    one GEMM batch ``L = d * B + b``. A transposed fusion there permutes whole batches without
    changing any value, which no aggregate check would see -- so each batch is compared to its own
    single-batch run.

    Args:
        B: The batch.
        N: The token extent.
    """
    D = 64
    kw = _weights(B, N, D, bias=True)
    got = T.trimul_autotuned(**kw, direction="outgoing", eps=1e-5, select="heuristic")
    ref = _oracle(kw, "outgoing")
    for bi in range(B):
        assert_elementwise(
            got[bi].float(), ref[bi], _e2e_bound(ref[bi]), what=f"batch {bi} of {B} at N={N}"
        )


# ── the refusals ───────────────────────────────────────────────────────────────────────────────


@requires_sm90
@TRIMUL_AUTOTUNE.parametrize_unsupported("N", "D", "direction")
def test_unsupported_raises(N, D, direction, expected_error, expected_match):
    """Every declared unsupported shape is REFUSED at the dispatcher's own front door.

    Not an `xfail`: an xfail'd assertion cannot tell "refused" from "answered wrongly", so a
    dispatcher that returned a plausible tensor for an unaligned shape would satisfy it. And not an
    `assert`: ``python -O`` strips those, and a stripped floor here is a kernel-level refusal three
    frames down instead of a sentence naming the constraint.

    Args:
        N, D, direction: The shape and direction under test.
        expected_error, expected_match: Injected by the matrix from the declared region.
    """
    kw = _weights(1, N if N % 8 == 0 else 8, D if D % 8 == 0 else 8, bias=False)
    # Rebuild x at the faulty extents, leaving every weight consistent with them, so the FIRST
    # check the front door reaches is the one this region declares.
    kw["x"] = torch.empty(1, device="cuda", dtype=torch.bfloat16).as_strided(
        (1, N, N, D), (0, 0, 0, 0)
    )
    kw["norm_in_w"] = torch.randn(D, device="cuda", dtype=torch.float32)
    kw["norm_out_w"] = torch.randn(D, device="cuda", dtype=torch.float32)
    kw["p_in_w"] = torch.randn(2 * D, D, device="cuda", dtype=torch.bfloat16)
    kw["g_in_w"] = torch.randn(2 * D, D, device="cuda", dtype=torch.bfloat16)
    kw["p_out_w"] = torch.randn(D, D, device="cuda", dtype=torch.bfloat16)
    kw["g_out_w"] = torch.randn(D, D, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(expected_error, match=expected_match):
        T.trimul_autotuned(**kw, direction=direction, eps=1e-5, select="heuristic")


@requires_sm90
@matrix_exempt("the subject is the ONE combo no front door in this package expresses")
@pytest.mark.parametrize("variant", ["dual_gated_gemm__gemm__layernorm__gemm_hadamard"])
def test_inexpressible_combo_says_so_by_name(variant):
    """The one combo this package cannot express refuses by NAME, and says the gap is structural.

    It wants a two-activation GEMM with NO LayerNorm on either arm.
    `layernorm_dual_gated_gemm` deliberately does not port the flag that would give it one, and
    `dual_gated_gemm` takes a single activation -- so closing it is a kernel change, not a pending
    port. The message has to say so, because "not implemented" reads as "wait for it". It is not
    in the swept grid either, so only an explicit ``_config`` reaches it.

    Args:
        variant: The combo to force.
    """
    kw = _weights(1, 64, 128, bias=True)
    cfg = T.TriMulConfig(variant, front_v="alg_fold", back_v="alg_fold")
    with pytest.raises(NotImplementedError, match=r"structural rather than pending"):
        T.trimul_autotuned(**kw, direction="outgoing", eps=1e-5, _config=T._as_autotune_config(cfg))


@requires_sm90
@matrix_exempt("the subject is the SELECTION front door's own refusal, which no axis names")
def test_unknown_select_raises():
    """An unrecognised ``select`` is refused rather than silently taking one of the three paths."""
    kw = _weights(1, 64, 128, bias=False)
    with pytest.raises(ValueError, match=r"select must be"):
        T.trimul_autotuned(**kw, select="hueristic")


@requires_sm90
@matrix_exempt(
    "the subject is the SELECTION path itself -- that `select='autotune'` runs at all -- which is "
    "a property of the dispatcher and not of any shape axis"
)
def test_the_measured_selection_path_runs_and_agrees_with_the_oracle():
    """``select="autotune"`` reaches a combo and computes the right answer.

    **This is a regression test for a TypeError that made the option unusable at every shape.**
    `_trimul_config_is_valid` re-wrapped its argument as ``AutotuneConfig(**config)`` while the
    tuner passes an `AutotuneConfig`, so ``**`` refused it on the FIRST candidate of every sweep.
    ``select="autotune"`` raised, and `trimul_freeze` raised with it, because freezing begins with
    a measured call.

    **Nothing caught it, and the reason is the whole point of this test.** Every other numerical
    cell here pins a combo with ``_config=``, which short-circuits inside the tuner BEFORE the
    validity gate, or takes ``select="heuristic"``, which never enters the tuner at all. The
    validity callback had therefore never executed, in any test, on any shape.

    ``CPO_AUTOTUNE=0`` keeps this cheap and does NOT weaken it: the tuner then takes the first
    admissible config instead of measuring, but it reaches that config through the SAME
    `candidates` -> `admissible` -> validity chain that was broken, so the defect is still in the
    path. Measuring instead would compile every candidate to assert the same thing.

    The result is checked against the fp32 oracle rather than against another selection's output:
    two combos may each be correct and still differ by more than `_e2e_bound` allows from each
    other, so a combo-to-combo comparison is not the bar.
    """
    monkey = pytest.MonkeyPatch()
    monkey.setenv("CPO_AUTOTUNE", "0")
    try:
        kw = _weights(1, 128, 128, bias=True)
        got = T.trimul_autotuned(**kw, direction="outgoing", eps=1e-5, select="autotune")
        ref = _oracle(kw, "outgoing")
        assert got.shape == ref.shape and got.dtype == kw["x"].dtype
        assert_elementwise(got.float(), ref, _e2e_bound(ref), what="select='autotune'")
    finally:
        monkey.undo()


@requires_sm90
@matrix_exempt(
    "the subject is the FREEZE entry point -- that it resolves a combo and binds it -- which no "
    "shape or dtype axis varies"
)
def test_freezing_binds_a_combo_and_the_frozen_call_reproduces_it():
    """`trimul_freeze` resolves a pick, exposes it, and the bound callable reruns exactly it.

    **Freeze is the shape of entry point that ships broken**: it is expensive to call, so nothing
    calls it, so nothing notices when the accessor it depends on moves. This tree has already paid
    for that once -- ``.best_config`` was read off the decorated wrapper, which only ever carries
    ``.autotuner`` and ``.axes``, so the first call raised `AttributeError` after passing a full
    correctness suite and a byte-identity sweep. This test exists so the FIX is exercised, not
    merely written.

    Three things are asserted, and the third is the one a smoke test would miss: the frozen call
    must reproduce the SAME combo, not merely produce a plausible tensor. A `frozen_call` that
    dropped its ``_config`` and re-dispatched through the heuristic would return a correct answer
    from the wrong kernel, which is exactly the silent form of this failure.

    ``CPO_AUTOTUNE=0`` costs one compile rather than a sweep; see
    `test_the_measured_selection_path_runs_and_agrees_with_the_oracle` for why that is not a
    weaker test of the accessor.
    """
    monkey = pytest.MonkeyPatch()
    monkey.setenv("CPO_AUTOTUNE", "0")
    try:
        kw = _weights(1, 128, 128, bias=True)
        x = kw.pop("x")
        frozen = T.trimul_freeze(x, **kw, direction="outgoing", eps=1e-5)
        assert isinstance(frozen.config, T.AutotuneConfig), (
            f"freeze must expose its pick on `.config`; got {frozen.config!r}"
        )
        got = frozen(x)
        ref = _oracle({**kw, "x": x}, "outgoing")
        assert_elementwise(got.float(), ref, _e2e_bound(ref), what="frozen call")
        # The frozen combo, re-run through the ordinary front door as an explicit pin, must give
        # BITWISE the same answer -- which is what says the frozen callable dispatched to its own
        # pick rather than quietly re-selecting.
        pinned = T.trimul_autotuned(x, **kw, direction="outgoing", eps=1e-5, _config=frozen.config)
        assert torch.equal(got, pinned), (
            f"the frozen callable did not run its own pick {frozen.config}: its output differs "
            f"from the same config dispatched explicitly"
        )
    finally:
        monkey.undo()


@requires_sm90
@TRIMUL_AUTOTUNE.parametrize(
    "N",
    only={"N": (2048, 4096)},
    because=(
        "the `workflow_N_token` facet is the whole subject: this is the only place the dispatcher "
        "is exercised at the token counts the A2A-fused workflow actually runs, and every other "
        "cell in this file is 4x-24x below the smallest of them"
    ),
)
def test_the_dispatcher_runs_at_the_workflow_token_counts(N):
    """The dispatcher selects and runs at the workflow's own ``N_token``, deterministically.

    **Why this exists.** Every other numerical cell in this file runs at ``N_token <= 512``, while
    the workflow runs 2048-12288. Combo selection is a function of ``(B, N, D)``: `_prune` and
    `_trimul_heuristic_config` both branch on total work ``B*N**2*D`` and on per-kernel validity
    floors, so a shape 4x-24x below the smallest production cell can take a DIFFERENT branch than
    anything production ever sees. Until this test the whole heuristic tree was argued at sizes it
    is not used at.

    **What it checks, and what it deliberately does not.** It asserts the pick is runnable, that
    the output has the right shape and dtype, that it is finite, and that two runs on identical
    operands agree BITWISE. It does NOT compare against `_oracle`, and that omission is arithmetic
    rather than laziness: the fp32 reference materializes ``xn``, a 2D-wide ``p_in`` and ``g_in``,
    and the token-pair einsum's output, which at ``N=2048, D=128`` is already ~20 GB of
    intermediates and at ``N=4096`` about 80 GB. The numerical claim is carried by
    `test_matches_the_reference` at small ``N``, where the whole output can be checked; what only a
    production-sized cell can carry is that the chosen combo runs there at all.

    **Bitwise determinism is the non-vacuous part.** A persistent kernel reuses one statistics
    scratch across many work tiles, and the races that produces appear only when the grid is deep
    enough to recycle stages -- which is exactly what a production token count does and a 512-token
    cell does not. This is the shape at which such a race would first be visible.

    Skipping is by runtime `torch.OutOfMemoryError` only. A memory-estimate gate would have to model
    every intermediate the chosen combo allocates, and it would be wrong the moment the heuristic
    picked a different one -- which is precisely the freedom this test exists to exercise.

    Args:
        N: The token extent, from the `workflow_N_token` facet.
    """
    D = 128  # the smallest workflow feature width: N**2 * D * 2 bytes is 1.07 GB at 2048, 4.3 at 4096
    try:
        kw = _weights(1, N, D, bias=True)
        got = T.trimul_autotuned(**kw, direction="outgoing", eps=1e-5, select="heuristic")
        torch.cuda.synchronize()
        assert got.shape == (1, N, N, D), f"expected (1, {N}, {N}, {D}), got {tuple(got.shape)}"
        assert got.dtype == kw["x"].dtype
        assert torch.isfinite(got.float()).all(), (
            "the workflow-sized output contains non-finite values"
        )
        again = T.trimul_autotuned(**kw, direction="outgoing", eps=1e-5, select="heuristic")
        torch.cuda.synchronize()
        n_bad = int((got != again).sum())
        assert_bitwise(
            got,
            again,
            what=f"the dispatcher is not deterministic at N_token={N}: {n_bad} of {got.numel()} elements differ between two runs on identical operands. At this token count the grid recycles per-CTA scratch across many work tiles, which is where such a race appears.",
        )
    except torch.OutOfMemoryError:
        pytest.skip(f"N_token={N} at D={D} does not fit on this device")
    finally:
        torch.cuda.empty_cache()
