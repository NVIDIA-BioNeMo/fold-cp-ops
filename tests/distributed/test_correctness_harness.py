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
"""Tests for ``tests.distributed.correctness_harness`` -- the error-distribution diagnostic.

**The subject is a DIAGNOSTIC, so the tests ask whether it points at the right place**, not whether
it agrees with some bar. Every case here plants a known defect in a known cell and asserts the
histogram localizes it: the outlier count sees it, the worst-row index names it, and the pooled
``rel_l2`` -- deliberately -- does not.

That last one is the test worth having. ``rel_l2`` is the field the upstream gated on, and this
file's job is to demonstrate, with a number, that a one-row corruption can sit inside a plausible
``rel_l2`` while the per-row view reports it. It is the same demonstration that put
`fold_cp_ops.testing.numerics` in place, run against the harness that motivated it.

CPU-only and process-local: everything is `torch.randn` plus arithmetic, no CUDA, no process group,
no collective. It lives under ``tests/distributed/`` because its subject does.
"""

import pytest
import torch

from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    matrix_exempt,
    no_unsupported,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt

from tests.distributed.correctness_harness import (
    ErrorHistogram,
    compute_error_histogram,
    compute_error_histogram_blocked,
)

HARNESS = KernelMatrix(
    kernel="correctness_harness",
    axes=(
        Axis(
            name="shape",
            domain=(
                "any tensor shape of rank >= 4, since the reduction is flatten(0, 2).norm(dim=-1) "
                "and therefore needs three leading axes plus a feature axis. The pool spans the "
                "two ranks the callers actually pass -- the 4-D (B, N, N, D) the docstring is "
                "written for, and the 5-D (cp, Dloc, B, N_loc, N) recv the A2A stores produce -- "
                "because the row decode is derived from the first three extents and a pool of one "
                "rank could not tell a rank-agnostic reduction from a 4-D-only one"
            ),
            values=(
                (1, 4, 4, 8),  # smallest 4-D
                (2, 8, 8, 16),  # 4-D, every extent distinct from its neighbours
                (2, 3, 1, 8, 16),  # 5-D recv, B=1 -- the shipped back-store case
                (2, 2, 2, 4, 8),  # 5-D recv, every leading extent > 1
            ),
            facets={
                "four_d": lambda s: len(s) == 4,
                "five_d": lambda s: len(s) == 5,
                "unit_extent": lambda s: 1 in s,
                "all_multi": lambda s: 1 not in s,
            },
        ),
    ),
    computes=computes_nothing_numeric(
        because=(
            "the subject REPORTS an error distribution rather than computing a value to be "
            "compared against a reference. Its own output is the comparison, so there is no "
            "downstream tensor whose element distribution could hide anything -- the tests here "
            "plant a defect and assert the report finds it"
        )
    ),
    unsupported=no_unsupported(
        because=(
            "ENUMERATED, not sampled: the module has one public function and exactly one raise, "
            "the shape-mismatch assert. That fires on MALFORMED input -- two tensors of different "
            "shapes -- rather than on any combination of declared axis values, and every shape in "
            "the pool is used for both operands. It is covered directly by "
            "test_a_reference_of_a_different_shape_is_refused"
        )
    ),
)


def _pair(shape, *, seed=0):
    """A (got, ref) pair that is EXACTLY equal -- the baseline every defect is planted into.

    Exact rather than near-exact on purpose: a test that plants one bad row into an already-noisy
    pair cannot attribute the outlier it then measures.

    Args:
        shape: The tensor shape, rank >= 4.
        seed: Seeds `torch.manual_seed` so a failure is reproducible from the test id alone.

    Returns:
        ``(got, ref)``, two fp32 tensors that compare equal elementwise.
    """
    torch.manual_seed(seed)
    ref = torch.randn(*shape)
    return ref.clone(), ref


@HARNESS.parametrize("shape")
@numeric_exempt(
    "asserts the fields of a DIAGNOSTIC report about a comparison, not the result of one. The "
    "tensors here are constructed equal by the helper rather than computed by a kernel"
)
def test_an_exact_match_reports_no_error_anywhere(shape):
    """Two identical tensors produce a zero report -- every field, not just the headline.

    A histogram that reported a small nonzero error on an exact match would make every later
    threshold meaningless, and the failure would be invisible: a tiny nonzero passes every bar.
    """
    got, ref = _pair(shape)
    h = compute_error_histogram(got, ref)
    assert h.rel_l2 == 0.0, f"exact match reported rel_l2={h.rel_l2}"
    assert h.max_abs == 0.0, f"exact match reported max_abs={h.max_abs}"
    assert h.n_over_thresh == 0, f"exact match reported {h.n_over_thresh} elements over threshold"
    assert h.n_outlier_rows == 0, f"exact match reported {h.n_outlier_rows} outlier rows"


@HARNESS.parametrize("shape")
@numeric_exempt("asserts that a planted defect is LOCALIZED by the report, not a computed value")
def test_one_corrupted_row_is_reported_as_exactly_one_outlier_row(shape):
    """Collapsing a single token row to zero is reported as exactly one outlier row.

    This is the defect class the per-row view exists for: a dropped or unwritten token row, which
    is what an A2A scatter store produces when one peer's tile never lands. The count must be
    exactly one -- a count of zero misses it, and a count above one means the threshold is flagging
    healthy rows and the signal is unusable.
    """
    got, ref = _pair(shape)
    got = got.clone()
    # A "row" is one LAST-AXIS vector, so reshape over EVERY leading axis. Measured the hard way:
    # `flatten(0, 2)[0]` looks like the harness's own grouping and is not -- at 5-D it selects a
    # (N_loc, N) slab, i.e. N_loc rows at once, and the test reported 4 outliers for one planted
    # defect. It coincides with a single row only at 4-D, which is why a 4-D-only pool would have
    # passed against this spelling and taught nothing.
    got.reshape(-1, shape[-1])[0] = 0.0
    h = compute_error_histogram(got, ref)
    assert h.n_outlier_rows == 1, (
        f"zeroing one row of {shape} was reported as {h.n_outlier_rows} outlier rows; {h.summary()}"
    )


@matrix_exempt(
    "the subject is the CONTRAST between a pooled scalar and a per-row count at ONE deliberately "
    "large shape -- the demonstration needs many rows so the corrupted one is a small fraction, "
    "which is the property the matrix's small pool shapes cannot exhibit"
)
@numeric_exempt("asserts the RELATIONSHIP between two reported metrics, not a computed value")
def test_a_pooled_rel_l2_hides_the_row_the_per_row_view_finds():
    """One collapsed row out of 4096 leaves ``rel_l2`` small while ``n_outlier_rows`` reports it.

    **This is the argument for the element-wise rule, run as a test rather than asserted in prose.**
    The upstream gated on ``rel_l2 < 2e-2``; a single corrupted row among many contributes about
    ``sqrt(1/n_rows)`` of the norm, so at 4096 rows it moves the pooled metric by ~1.6% -- inside
    that bar. The per-row count sees it immediately.

    The numbers are asserted, not narrated: if a future change made ``rel_l2`` sensitive enough to
    catch this, or made the row count miss it, this test says so.
    """
    torch.manual_seed(0)
    ref = torch.randn(1, 64, 64, 32)  # 4096 rows
    got = ref.clone()
    got.reshape(-1, 32)[0] = 0.0
    h = compute_error_histogram(got, ref)
    assert h.rel_l2 < 2e-2, (
        f"the demonstration no longer holds: rel_l2={h.rel_l2:.3e} is OUTSIDE the upstream's 2e-2 "
        "bar, so this shape no longer shows a pooled metric hiding a row. Re-derive the row count "
        "at which it does, rather than deleting the test"
    )
    assert h.n_outlier_rows == 1, (
        f"the per-row view missed the collapsed row (n_outlier_rows={h.n_outlier_rows}); with "
        f"rel_l2={h.rel_l2:.3e} inside the bar, NOTHING would have caught it. {h.summary()}"
    )


@matrix_exempt(
    "the subject is the worst-row DECODE, which is documented as meaningful only for the 4-D "
    "(B, N, N, D) shape -- sweeping the 5-D pool values would assert a coordinate the module's own "
    "docstring says is not a coordinate there"
)
@numeric_exempt("asserts an index, not a computed value")
@pytest.mark.parametrize(
    "bad", [(0, 0, 0), (1, 3, 5), (1, 7, 7)], ids=["first", "interior", "last"]
)
def test_the_worst_row_index_names_the_row_that_was_corrupted(bad):
    """``worst_row_index`` is the ``(b, i, j)`` of the row actually corrupted, at 4-D.

    Three positions because the decode is three integer divisions: a first-row-only test passes
    against a decode that always returns ``(0, 0, 0)``, and a last-row test passes against one that
    is off by a constant. The interior case pins the two divisions independently.
    """
    torch.manual_seed(0)
    ref = torch.randn(2, 8, 8, 16)
    got = ref.clone()
    got[bad] = 0.0
    h = compute_error_histogram(got, ref)
    assert h.worst_row_index == bad, (
        f"corrupted row {bad} but the report names {h.worst_row_index}; a diagnostic that points "
        f"at the wrong cell is worse than none. {h.summary()}"
    )


@matrix_exempt("asserts a refusal on MALFORMED input, which is not a combination of pool values")
@numeric_exempt("asserts a refusal, not a computed value")
def test_a_reference_of_a_different_shape_is_refused():
    """A shape mismatch raises rather than broadcasting into a plausible-looking report.

    Broadcasting is the hazard: ``(2, 8, 8, 16)`` against ``(2, 8, 8, 1)`` would subtract fine and
    produce a full histogram for a comparison nobody asked for, and every number in it would look
    ordinary.
    """
    got = torch.randn(2, 8, 8, 16)
    ref = torch.randn(2, 8, 8, 1)
    with pytest.raises(AssertionError, match="shape mismatch"):
        compute_error_histogram(got, ref)


@matrix_exempt("asserts the report's own string rendering, which does not vary with shape")
@numeric_exempt("asserts a message's content, not a computed value")
def test_the_summary_line_carries_every_field_a_failure_needs():
    """``summary()`` names the worst row and the outlier count, not just the pooled scalar.

    The summary is what a failing test prints, so a summary that showed only ``rel_L2`` would leave
    the reader with exactly the number this module exists to supplement.
    """
    torch.manual_seed(0)
    ref = torch.randn(1, 4, 4, 8)
    got = ref.clone()
    got.reshape(-1, 8)[3] = 0.0
    line = compute_error_histogram(got, ref).summary()
    for field in ("rel_L2", "outlier_rows", "worst@", "row_med", "row_max", "p99"):
        assert field in line, f"summary() omits {field!r}: {line}"


@matrix_exempt("asserts the dataclass's field set, a property of the type rather than of a shape")
@numeric_exempt("inspects a type's fields, not a computed value")
def test_every_declared_field_is_populated():
    """No field is left at a default -- a partially-filled report reads as a zero measurement.

    `ErrorHistogram` is a plain dataclass with no defaults, so a missing field is a TypeError at
    construction. What this checks is the softer failure: a field populated with ``None`` or NaN by
    a future edit, which formats without complaint and reads as "measured, and it was nothing".
    """
    torch.manual_seed(0)
    ref = torch.randn(1, 4, 4, 8)
    h = compute_error_histogram(ref.clone(), ref)
    for name in ErrorHistogram.__dataclass_fields__:
        value = getattr(h, name)
        assert value is not None, f"field {name} is None"
        if isinstance(value, float):
            assert value == value, f"field {name} is NaN"  # noqa: PLR0124 -- NaN self-comparison


@pytest.mark.parametrize("n_blocks", [1, 3, 7, 1200])
@matrix_exempt(
    "asserts that a STREAMED comparison equals the whole-tensor one for any block count. The "
    "property is an algebraic identity over a partition, not something that varies with a declared "
    "extent -- parametrizing the pool would re-assert the same identity at every shape"
)
@numeric_exempt(
    "compares two REPORTS of the same comparison against each other, not a computed tensor "
    "against a reference"
)
def test_the_streamed_report_equals_the_whole_tensor_one(n_blocks):
    """Streaming the reference changes the memory, not a single reported number.

    This is the property the streamed path exists to have, and it is the one a reader cannot check
    by inspection: the front reference is ``2*D*B*N^2`` elements -- ``cp`` CANCELS, so a bigger mesh
    does not shrink it -- and assembling it asked for a single 33.79 GiB allocation beside an
    equally large recv. Consuming it per block removes that allocation, and this asserts the removal
    costs nothing: every field that gates a test must be BIT-equal or within fp accumulation noise,
    at block counts from one (degenerate: the whole tensor) to one row per block (degenerate: the
    finest possible split). A silent disagreement here would move a bar without anyone editing one.

    ``pct`` is deliberately NOT compared: percentiles are not decomposable over a partition, so the
    streamed path subsamples -- exactly as the whole-tensor path already does above 16M elements.
    """
    torch.manual_seed(7)
    B, N1, N2, F = 1, 40, 30, 32
    ref = torch.randn(B, N1, N2, F)
    got = ref.clone()
    got.view(-1)[517] += 3.0  # one corrupted ELEMENT -- the case a pooled scalar would average away
    got.view(B, -1, F)[0, 777, :] *= 4.0  # and one corrupted ROW, so the row fields are non-trivial
    whole = compute_error_histogram(got, ref)

    rows_got, rows_ref = got.reshape(-1, F), ref.reshape(-1, F)
    step = (rows_got.shape[0] + n_blocks - 1) // n_blocks
    streamed = compute_error_histogram_blocked(
        (
            (rows_got[i : i + step], rows_ref[i : i + step])
            for i in range(0, rows_got.shape[0], step)
        ),
        abs_thresh=whole.abs_thresh,
        row_dims=(B, N1, N2),
    )
    for field in (
        "max_abs",
        "mean_abs",
        "rel_l2",
        "n_over_thresh",
        "n_rows",
        "row_med",
        "row_max",
        "row_outlier_ratio",
        "n_outlier_rows",
        "worst_row_index",
    ):
        a, b = getattr(whole, field), getattr(streamed, field)
        if isinstance(a, (int, tuple)):
            assert a == b, f"{field}: whole={a!r} streamed={b!r} at n_blocks={n_blocks}"
        else:
            assert abs(a - b) <= 1e-5 * max(1.0, abs(a)), (
                f"{field}: whole={a!r} streamed={b!r} at n_blocks={n_blocks}"
            )


@matrix_exempt("asserts a refusal on an EMPTY stream, which is not a combination of pool values")
@numeric_exempt("asserts a raise, not a numerical comparison")
def test_an_empty_stream_is_refused_rather_than_reported_as_clean():
    """A stream that yields nothing RAISES instead of reporting a vacuous all-zero histogram.

    Without this the degenerate case is the dangerous one: zero elements produce zero error on every
    field, which is indistinguishable from a perfect result and passes every bar. A generator that
    silently yields nothing -- an empty peer list, a mis-derived block range -- would then read as a
    pass for a comparison that never happened.
    """
    with pytest.raises(AssertionError, match="EMPTY stream"):
        compute_error_histogram_blocked(iter([]), abs_thresh=1.0, row_dims=(1, 1, 1))


# --------------------------------------------------------------------------- #
# make_weights -- is the POOL potent? (D4)
# --------------------------------------------------------------------------- #
#: The bound `test_the_fused_chain_matches_the_fp32_oracle` compares against. Duplicated as literals
#: rather than imported because that module needs a live process group at import time, and this file
#: is CPU-only and process-local by design. A drift between the two is what `_E2E_ATOL`/`_E2E_RTOL`
#: appearing in a failure message is meant to make obvious.
_E2E_ATOL, _E2E_RTOL = 2e-2, 6e-2


@matrix_exempt(
    "the subject is the WEIGHT BUILDER's output distribution, not a kernel run over a shape "
    "combination -- there is no kernel here and the feature width is the builder's own argument"
)
@numeric_exempt(
    "measures the POOL's SENSITIVITY to a weight -- how far the oracle moves when an input is "
    "deleted -- not a computed tensor against a reference, so there is no element to shadow"
)
@pytest.mark.parametrize("D", [128, 256, 384, 512])
def test_the_weight_pool_can_actually_see_the_layernorm_bias(D):
    """Deleting the LayerNorm bias from `make_weights` must move the oracle past the e2e bound.

    Purpose
        Ask whether the pool can SEE the kernel, not whether the kernel is right. `make_weights`
        feeds every A2A correctness test, and `TriMulAutotuned` really does fuse both LayerNorm
        biases -- ``norm_in_b`` into `layernorm_fwd`, ``norm_out_b`` into
        `layernorm_dual_gated_gemm(norm_bias=)`. If the pool sets them to a value at which they
        contribute nothing, those code paths are untested while every test stays green.

    Semantics
        Computes the fp32 oracle twice on one input -- once with the pool's weights, once with both
        LayerNorm biases zeroed -- and counts the elements where the difference exceeds the SAME
        `tolerance_bound` the end-to-end test uses. That count is the fraction of the output on which
        a kernel ignoring the bias entirely would be CAUGHT.

        **This test used to fail, and that is why it exists.** The pool set ``norm_*_b =
        zeros(D)`` -- the additive identity -- so the two oracles agreed BITWISE: measured
        ``max_delta`` exactly ``0.0000`` and ``caught = 0.0%`` at ``B=1, N=256, D=128``. A fused
        kernel that never read the bias was indistinguishable from a correct one. With the affine
        off the identity the same measurement gives 86.7%. The ``0.5`` threshold below sits far under
        that, so this fails on a regression to the identity and not on ordinary resampling.

        The same argument covers the GAIN, which was ``ones(D)``: ``alg_fold`` folds it into the GEMM
        weight on the host, so at ``gain == 1`` the fold is a no-op copy. Zeroing the bias is the
        cheaper probe and a regression would revert both together, so only the bias is planted here.

    Args:
        D: feature width, from the four the A2A workflow declares. Small ``N`` below is deliberate --
            this file is CPU-only, and the property is a per-FEATURE one, so it does not need the
            workflow's token extent to show up.

    Raises:
        AssertionError: if fewer than half the output elements move past the bound -- i.e. the pool
            has drifted back toward a value at which the LayerNorm bias does not matter.
    """
    from fold_cp_ops.testing.numerics import tolerance_bound

    from tests.distributed.correctness_harness import global_oracle, make_weights

    B, N, eps = 1, 64, 1e-5
    w = make_weights(D, seed=20260826, device="cpu")
    g = torch.Generator(device="cpu").manual_seed(20260826 + 1)
    x = torch.randn(B, N, N, D, generator=g, dtype=torch.float32)

    kw = dict(direction="outgoing", mask_global=None, eps=eps, row_tile=None)
    ref = global_oracle(x, w, **kw).float()
    w_flat = {**w, "norm_in_b": torch.zeros(D), "norm_out_b": torch.zeros(D)}
    ref_flat = global_oracle(x, w_flat, **kw).float()

    delta = (ref - ref_flat).abs()
    caught = (delta > tolerance_bound(ref, atol=_E2E_ATOL, rtol=_E2E_RTOL)).float().mean().item()
    assert caught >= 0.5, (
        f"D={D}: zeroing the LayerNorm biases moves only {caught * 100:.1f}% of the output past the "
        f"e2e bound (atol={_E2E_ATOL}, rtol={_E2E_RTOL}); max delta {delta.max().item():.4g}. The "
        f"pool cannot see a term the fused workflow computes, so a kernel that dropped "
        f"norm_in_b/norm_out_b would pass every A2A correctness test. Check that "
        f"`make_weights` still draws the LayerNorm affine OFF the identity."
    )


@matrix_exempt(
    "the subject is the WEIGHT BUILDER's output distribution, not a kernel run over a shape "
    "combination -- there is no kernel here and the feature width is the builder's own argument"
)
@numeric_exempt(
    "measures the POOL's SENSITIVITY to a weight -- how far the oracle moves when an input is "
    "deleted -- not a computed tensor against a reference, so there is no element to shadow"
)
@pytest.mark.parametrize("family", ["in", "out"])
@pytest.mark.parametrize("D", [128, 512])
def test_the_weight_pool_can_actually_see_the_projection_biases(family, D):
    """Deleting a PROJECTION bias must move the oracle past the e2e bound, on both pairs.

    Purpose
        The companion to `test_the_weight_pool_can_actually_see_the_layernorm_bias`, for the four
        biases that are opt-in rather than always-on. All four are fused -- ``p_in_b``/``g_in_b``
        into the front store's interleaved ``mRowVecBroadcast``, ``p_out_b``/``g_out_b`` into
        `layernorm_dual_gated_gemm` as ``bp``/``bg`` -- so each needs a pool that can tell whether the
        kernel read it.

        Worth having separately from the LayerNorm one because the two pairs enter the chain at
        different points and could fail to register for different reasons: the front pair is added
        before the glu and then survives the A2A feature scatter, while the back pair is added after
        the token-pair einsum.

    Args:
        family: ``"in"`` sets ``p_in_b``/``g_in_b``, ``"out"`` sets ``p_out_b``/``g_out_b``.
        D: feature width; two of the four the workflow declares, since this is the slower of the two
            potency tests and the property does not vary across them.

    Raises:
        AssertionError: if fewer than half the output elements move past the bound -- the pool would
            then pass a kernel that ignored the bias entirely.
    """
    from fold_cp_ops.testing.numerics import tolerance_bound

    from tests.distributed.correctness_harness import global_oracle, make_weights

    B, N, eps = 1, 64, 1e-5
    keys = ("p_in_b", "g_in_b") if family == "in" else ("p_out_b", "g_out_b")
    w = make_weights(
        D, in_bias=(family == "in"), out_bias=(family == "out"), seed=20260826, device="cpu"
    )
    assert all(w[k] is not None for k in keys), f"{family} pool did not build {keys}"

    g = torch.Generator(device="cpu").manual_seed(20260826 + 1)
    x = torch.randn(B, N, N, D, generator=g, dtype=torch.float32)
    kw = dict(direction="outgoing", mask_global=None, eps=eps, row_tile=None)
    ref = global_oracle(x, w, **kw).float()
    ref_flat = global_oracle(x, {**w, **{k: None for k in keys}}, **kw).float()

    delta = (ref - ref_flat).abs()
    caught = (delta > tolerance_bound(ref, atol=_E2E_ATOL, rtol=_E2E_RTOL)).float().mean().item()
    assert caught >= 0.5, (
        f"D={D} family={family}: dropping {keys} moves only {caught * 100:.1f}% of the output past "
        f"the e2e bound; max delta {delta.max().item():.4g}. A kernel that ignored those biases "
        f"would pass the A2A correctness tests."
    )
