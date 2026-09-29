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

"""Tests for ``fold_cp_ops._internal.ln_prologue`` -- the fused LayerNorm prologue's four helpers.

Driven through ``_ln_prologue_kernel``, a harness that runs the production sequence with the WGMMA
and the pipeline removed, so a failure names the prologue rather than the GEMM around it.

**Two properties get exact gates, not tolerances.** The published statistics are compared against a
double-precision oracle at a tight bound, and the ZERO of a padded tail is compared with
``torch.equal``: the padded columns must be exactly 0.0, because their whole purpose is to
contribute exactly nothing to a product. A tolerance there would let a small non-zero through, and
a small non-zero times a large weight is not small.

**On the reference bound.** The normalize writes back in the ACTIVATION's dtype, so its output
carries one rounding the fp32 oracle does not. The tolerance below is derived from that -- half an
ulp of the output dtype at the output's own magnitude -- rather than picked.
"""

import pytest
import torch

from fold_cp_ops._internal.compile_time.ln_prologue_layout import auto_blk_k
from fold_cp_ops.testing.numerics import assert_elementwise

from tests._internal._ln_prologue_kernel import run_ln_prologue

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(
    _SM != 9, reason=f"the LayerNorm prologue needs sm_90; this GPU is sm_{_SM}0"
)


def _ref(x, weight, bias, eps=1e-5):
    """Double-precision LayerNorm oracle, returned in fp64.

    Args:
        x: ``(M, K)`` activation of any float dtype.
        weight: ``(K,)`` gain.
        bias: ``(K,)`` bias, or None.
        eps: The variance floor.

    Returns:
        The fp64 result. fp64 rather than fp32 so the oracle's own error is negligible beside the
        kernel's, which is what lets the tolerance below describe the KERNEL.
    """
    return torch.nn.functional.layer_norm(
        x.double(),
        (x.shape[-1],),
        weight.double(),
        bias.double() if bias is not None else None,
        eps,
    )


def _bound(ref, dtype):
    """The tolerance the fusion's extra rounding justifies: half an ulp of `dtype`, per element.

    Purpose
        States WHY the bound is what it is, so a future widening has to argue with the reason rather
        than with a number.

    Semantics
        The normalize computes in fp32 and stores in `dtype`, so each output carries one round-to-
        nearest of its own magnitude on top of the fp32 arithmetic. Half an ulp is that rounding's
        exact worst case; the ``2x`` factor covers the fp32 accumulation of ``Sum x`` / ``Sum x^2``
        underneath it, which is the only other error term.

    Args:
        ref: The fp64 oracle, whose magnitudes set the per-element ulp.
        dtype: The output dtype -- bf16 (8 mantissa bits) or fp16 (11).

    Returns:
        A fp64 tensor of per-element absolute tolerances, plus a floor so a near-zero output is not
        held to a zero tolerance.
    """
    mant = {torch.bfloat16: 8, torch.float16: 11}[dtype]
    ulp = ref.abs() * (2.0 ** -(mant - 1))
    return 2.0 * ulp + 1e-3


def _make(M, K, dtype, seed=0, with_bias=True):
    """Build one cell's operands.

    Args:
        M: Row extent. Must be a multiple of the harness's ``tile_m``.
        K: Feature extent, the LayerNorm axis.
        dtype: The activation's element type; a 16-bit type.
        seed: RNG seed, so a failure reproduces.
        with_bias: Whether to build a LayerNorm bias.

    Returns:
        ``(x, weight, bias)`` on CUDA. ``x`` is given a NON-ZERO per-row mean on purpose: a
        zero-mean input makes ``E[x^2] - mu^2`` agree with a centred sum even when the mean is
        computed wrongly, which is exactly how a variance bug hides.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=dtype, generator=g)
    x = x + torch.arange(M, device="cuda", dtype=dtype).reshape(-1, 1) * 0.01
    weight = torch.randn(K, device="cuda", dtype=torch.float32, generator=g)
    bias = torch.randn(K, device="cuda", dtype=torch.float32, generator=g) if with_bias else None
    return x, weight, bias


# ------------------------------------------------------------------ the whole prologue


@requires_sm90
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("with_bias", [False, True], ids=["gain_only", "gain_and_bias"])
@pytest.mark.parametrize(
    "M,K", [(128, 128), (256, 256), (512, 384), (128, 512)], ids=["k128", "k256", "k384", "k512"]
)
def test_the_prologue_reproduces_layer_norm(dtype, with_bias, M, K):
    """The staged, tiled, two-pass prologue equals ``F.layer_norm`` within one output rounding.

    Swept across the four TriMul feature widths because the k-tile COUNT is what the two passes
    walk: at K=128 there are two tiles and at K=512 there are eight, and a helper that carried its
    partials wrongly across tiles would pass the first and fail the last.
    """
    x, w, b = _make(M, K, dtype)
    out = run_ln_prologue(x, w, b, tile_k=auto_blk_k(K))
    ref = _ref(x, w, b)
    err = (out.double() - ref).abs()
    tol = _bound(ref, dtype)
    bad = (err > tol).sum().item()
    assert bad == 0, (
        f"{bad}/{err.numel()} elements outside the derived bound; worst "
        f"{(err / tol).max().item():.2f}x tolerance"
    )


@requires_sm90
@pytest.mark.parametrize("K", [128, 256, 384], ids=["k128", "k256", "k384"])
def test_the_published_statistics_are_the_rows_own_mean_and_reciprocal_deviation(K):
    """The scratch's two columns are ``mu`` and ``rstd``, in that order -- gated directly.

    The normalize cannot tell them apart: swapping the columns produces a smooth, plausible,
    entirely wrong result. Reading them out and comparing to a double-precision oracle is what makes
    the ORDER an asserted fact rather than a convention.
    """
    x, w, b = _make(256, K, torch.bfloat16)
    _, mu, rstd = run_ln_prologue(x, w, b, tile_k=auto_blk_k(K), want_stats=True)
    xd = x.double()
    mu_ref = xd.mean(-1)
    rstd_ref = 1.0 / (xd.var(-1, unbiased=False) + 1e-5).sqrt()
    assert_elementwise(
        mu.double(), mu_ref, 1e-2 * mu_ref.abs().max().clamp(min=1.0), what="published mu"
    )
    assert_elementwise(rstd.double(), rstd_ref, 1e-3 * rstd_ref.abs(), what="published rstd")


@requires_sm90
def test_a_non_zero_row_mean_is_actually_subtracted():
    """A large constant offset per row must vanish, which a dropped mean would not do.

    ``torch.randn`` has a near-zero mean, so a prologue that skipped the centring entirely would
    still look almost right on random data -- the defect this package has already been bitten by
    once. Here the offset is 50, far larger than the data, so a dropped mean is unmissable.
    """
    x, w, b = _make(128, 256, torch.bfloat16)
    offset = torch.arange(128, device="cuda", dtype=torch.bfloat16).reshape(-1, 1) + 50.0
    xo = x + offset
    out = run_ln_prologue(xo, w, b, tile_k=64)
    ref = _ref(xo, w, b)
    err = (out.double() - ref).abs()
    assert (err <= _bound(ref, torch.bfloat16)).all(), (
        f"worst {(err / _bound(ref, torch.bfloat16)).max().item():.2f}x tolerance -- a per-row "
        f"offset of ~50 did not cancel"
    )


# ------------------------------------------------------------------ the k-tile geometry


@requires_sm90
@pytest.mark.parametrize("tile_k", [64, 32, 16], ids=["blk64", "blk32", "blk16"])
def test_every_k_tile_width_computes_the_same_function(tile_k):
    """All three tiles reproduce the oracle -- they associate the sums differently, not wrongly.

    Note what this does NOT assert: that the three agree BITWISE. They must not be expected to --
    the tile sets the k-loop trip count, so it reassociates the fp32 accumulation, which is exactly
    why the tile is a declared parameter of the fused functor rather than an internal detail.
    """
    x, w, b = _make(128, 192, torch.bfloat16)
    out = run_ln_prologue(x, w, b, tile_k=tile_k)
    ref = _ref(x, w, b)
    err = (out.double() - ref).abs()
    assert (err <= _bound(ref, torch.bfloat16)).all()


@requires_sm90
def test_the_k_tile_width_is_a_numerical_parameter_not_only_a_performance_one():
    """Two tiles give two bit patterns on the same input -- the claim the functor's docstring makes.

    If this ever passes as "identical", the reassociation argument is wrong and the fused kernel's
    ``blk_k`` could be chosen freely at run time. It is asserted here so the claim is measured
    rather than assumed.
    """
    x, w, b = _make(128, 512, torch.bfloat16)
    a = run_ln_prologue(x, w, b, tile_k=64)
    c = run_ln_prologue(x, w, b, tile_k=16)
    assert not torch.equal(a, c), (
        "blk_k=64 and blk_k=16 produced identical bits. Either the reassociation this package "
        "documents does not happen, or the harness is not varying what it thinks it is."
    )


@requires_sm90
def test_the_padded_tail_is_exactly_zero_so_it_contracts_to_nothing():
    """Columns past the true feature width are staged as an EXACT 0.0, gated with ``torch.equal``.

    This is the padded-K path's whole correctness argument: the loader zero-fills the activation
    past ``K``, the staging loop zeroes the gain there, and ``(0 - mu) * rstd * 0 + 0`` is exactly
    zero -- so the padded column contributes exactly nothing to the product. A tolerance here would
    accept a small non-zero, and a small non-zero times a large weight is not small.

    Driven by giving the harness a ``gemm_k`` larger than the real ``K`` and reading the statistics:
    the normalizer must still be the TRUE width, so the mean must not be diluted by the pad.
    """
    K = 128
    x, w, b = _make(128, K, torch.bfloat16)
    _, mu, _ = run_ln_prologue(x, w, b, tile_k=64, gemm_k=256, want_stats=True)
    assert_elementwise(
        mu.double(),
        x.double().mean(-1),
        1e-2,
        what="row mean under a padded contraction (a diluted mean means the normalizer used the "
        "padded extent, not the TRUE feature width)",
    )
    out_pad = run_ln_prologue(x, w, b, tile_k=64, gemm_k=256)
    out_exact = run_ln_prologue(x, w, b, tile_k=64, gemm_k=128)
    assert torch.equal(out_pad, out_exact), (
        "padding the contraction extent changed the result; the pad must contribute exactly zero"
    )


# ------------------------------------------------------------------ the partition


@requires_sm90
@pytest.mark.parametrize(
    "tile_m,num_threads",
    [(128, 256), (128, 128), (64, 128), (256, 256)],
    ids=["2wg", "1wg", "narrow", "tall"],
)
def test_a_thread_rescales_the_rows_whose_statistics_it_helped_compute(tile_m, num_threads):
    """One partition drives both passes, across every cooperating-thread geometry.

    The failure this guards is specific and silent: if the reduction and the normalize derived their
    partitions independently, a row could be rescaled with another row's ``(mu, rstd)``. The result
    stays smooth and finite, so only a value comparison catches it -- and only if the rows differ,
    which is why ``_make`` gives each row its own mean.
    """
    x, w, b = _make(512, 256, torch.bfloat16)
    out = run_ln_prologue(x, w, b, tile_m=tile_m, tile_k=64, num_threads=num_threads)
    ref = _ref(x, w, b)
    err = (out.double() - ref).abs()
    assert (err <= _bound(ref, torch.bfloat16)).all(), (
        f"worst {(err / _bound(ref, torch.bfloat16)).max().item():.2f}x tolerance at "
        f"tile_m={tile_m}, num_threads={num_threads}"
    )


@requires_sm90
def test_omitting_the_bias_is_not_the_same_as_passing_zeros_at_the_kernel_level():
    """A kernel built without a bias and one given a zero bias agree in VALUE, not in construction.

    The values must match: adding zero changes nothing. What differs is that the first compiled the
    add out entirely -- which is the property that makes ``mBias=None`` a compile key rather than a
    runtime branch. The value check is what proves the pruned path did not also prune something it
    should have kept.
    """
    x, w, _ = _make(128, 256, torch.bfloat16)
    zeros = torch.zeros(256, device="cuda", dtype=torch.float32)
    assert torch.equal(
        run_ln_prologue(x, w, None, tile_k=64), run_ln_prologue(x, w, zeros, tile_k=64)
    )


def test_the_register_row_map_covers_every_row_of_the_tile_exactly_once():
    """Over 256 threads the map enumerates rows 0..127 once each, and nothing outside.

    This is the property the register-source reduction rests on, and the reason its callers refuse
    every tile but ``(128, 128)``. Each thread owns two rows -- ``base`` and ``base + RS_ROW_STRIDE``
    -- and only the owner lane of each 4-lane quad writes them, so the rows actually STORED are the
    owners' pairs. If that set is not exactly ``range(128)``:

    * a row covered TWICE is written by two threads with different partial sums, and the survivor
      is whichever store lands last -- a non-deterministic wrong answer;
    * a row covered ZERO times keeps whatever the scratch held from the previous work tile.

    Neither raises. Both are silent, which is why the map is pinned here rather than trusted.

    The map is EMPIRICALLY derived (probed by encoding the ``(m, k)`` coordinate into the staged
    tile and reading back the register slots), so it is exactly the kind of constant that must not
    be re-derived by hand in a second place -- hence one shared function with this test, rather
    than a copy per kernel.
    """
    from fold_cp_ops._internal.ln_prologue import RS_ROW_STRIDE, rs_owned_rows

    written = []
    for tidx in range(256):
        base, is_owner = rs_owned_rows(tidx)
        if is_owner:
            written.extend((base, base + RS_ROW_STRIDE))

    assert sorted(written) == list(range(128)), (
        "the owner lanes must write rows 0..127 exactly once; got "
        f"{len(written)} writes covering {len(set(written))} distinct rows"
    )
    assert len(written) == len(set(written)), "a row is written by two threads"


def test_only_one_lane_per_quad_writes_and_the_quad_shares_its_rows():
    """The four lanes of a quad own the SAME two rows and exactly one of them stores.

    That pairing is what makes a ``RS_QUAD_LANES``-wide warp reduction the right way to complete a
    row's K-sum: the quad's lanes hold disjoint K of one row, so reducing across them finishes it.
    If the lanes of a quad mapped to DIFFERENT rows, that reduction would silently mix two rows'
    statistics -- and the result would still look like a plausible LayerNorm.
    """
    from fold_cp_ops._internal.ln_prologue import RS_QUAD_LANES, rs_owned_rows

    for quad_start in range(0, 256, RS_QUAD_LANES):
        bases = {rs_owned_rows(quad_start + lane)[0] for lane in range(RS_QUAD_LANES)}
        owners = [rs_owned_rows(quad_start + lane)[1] for lane in range(RS_QUAD_LANES)]
        assert len(bases) == 1, f"quad at {quad_start} spans rows {bases}, must share one"
        assert sum(owners) == 1, f"quad at {quad_start} has {sum(owners)} owners, must have 1"


def test_the_two_warpgroups_own_disjoint_halves_of_the_tile():
    """Warpgroup 0 owns rows 0..63 and warpgroup 1 owns 64..127, with no overlap.

    The leading ``t // 128`` term is what separates them. Reducing `tidx` to a warpgroup-LOCAL
    index before calling -- an easy thing to do when a caller already has one in hand -- collapses
    both warpgroups onto rows 0..63, so half the tile is never written and the other half is
    written twice. This pins the global-index requirement stated in the docstring.
    """
    from fold_cp_ops._internal.ln_prologue import RS_ROW_STRIDE, rs_owned_rows

    halves = []
    for wg in (0, 1):
        rows = set()
        for lane in range(128):
            base, is_owner = rs_owned_rows(wg * 128 + lane)
            if is_owner:
                rows.update((base, base + RS_ROW_STRIDE))
        halves.append(rows)
    assert halves[0] == set(range(64)), f"warpgroup 0 owns {sorted(halves[0])[:4]}..."
    assert halves[1] == set(range(64, 128)), f"warpgroup 1 owns {sorted(halves[1])[:4]}..."
    assert not (halves[0] & halves[1])
