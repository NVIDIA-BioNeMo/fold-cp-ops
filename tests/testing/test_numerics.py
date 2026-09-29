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
"""Tests for ``fold_cp_ops.testing.numerics`` -- the gates every kernel test is judged by.

A gate that cannot fail is worse than no gate: it reads as evidence. So most of these tests feed
a *deliberately wrong* tensor and assert the gate catches it, rather than only checking that a
correct one passes.

CPU-only and GPU-free: these are numerical-analysis properties, not kernel behaviour.
"""

import re

import pytest
import torch

from fold_cp_ops.testing import numerics
from fold_cp_ops.testing.numeric_guard import numeric_exempt

from fold_cp_ops.testing.numerics import (
    alg_fold_error_bound,
    alg_fold_sigmoid_error_bound,
    alg_fold_xgate_error_bound,
    assert_bitwise,
    assert_elementwise,
    assert_gemm_close,
    assert_gemm_exact,
    assert_written,
    exact_int_bound,
    fused_ln_gated_error_bound,
    gated_error_bound,
    gemm_error_bound,
    gemm_reference,
    integer_operands,
    max_exact_operand,
    reduction_error_bound,
    reduction_reference,
)


# ── the exactness arithmetic the strongest gate rests on ──────────────────────────────────────
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.float16, 2048),
        (torch.bfloat16, 256),
        (torch.float32, 2**24),
        (torch.float8_e4m3fn, 16),
        (torch.float8_e5m2, 8),
    ],
)
def test_exact_int_bound_matches_what_the_format_actually_represents(dtype, expected):
    """The table is right, checked against the hardware rather than against itself.

    ``exact_int_bound`` is a lookup table, and a wrong entry silently invalidates every
    integer-exact assertion downstream. So round-trip the boundary through the real dtype: the
    bound itself must survive, and ``bound + 1`` must NOT (it is the first integer that rounds).
    """
    assert exact_int_bound(dtype) == expected
    b = float(expected)
    assert float(torch.tensor([b], dtype=dtype).double()) == b, "the bound itself must be exact"
    if dtype is not torch.float32:  # fp32's bound+1 is not representable in a python float test
        assert float(torch.tensor([b + 1], dtype=dtype).double()) != b + 1, (
            "bound + 1 must NOT be exact, or the bound is understated"
        )


def test_exact_int_bound_refuses_an_unknown_dtype():
    """An unlisted dtype raises rather than defaulting -- a guessed precision is a silent lie."""
    with pytest.raises(KeyError, match=r"no precision known"):
        exact_int_bound(torch.int32)


@pytest.mark.parametrize("K", [1, 8, 128, 1024, 4096, 65536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_max_exact_operand_keeps_every_partial_sum_inside_fp32(K, dtype):
    """``K * P**2 < 2**24`` holds at the returned P, and fails at P + 1 unless the dtype capped it.

    This is the inequality the whole integer-exact argument depends on, so it is asserted directly
    rather than trusted from the derivation.
    """
    p = max_exact_operand(K, dtype)
    assert p >= 1, f"K={K} left no usable operand bound at {dtype}"
    assert K * p * p < 2**24
    if p < exact_int_bound(dtype):  # not dtype-capped, so p is the true maximum
        assert K * (p + 1) * (p + 1) >= 2**24, "p is not maximal"


def test_max_exact_operand_is_capped_by_the_dtype_not_only_by_K():
    """At small K the format, not the accumulator, is the binding constraint.

    bf16 represents integers only to 256, so a K=1 bf16 operand is capped there even though the
    fp32 accumulator would tolerate 4096.
    """
    assert max_exact_operand(1, torch.bfloat16) == 256
    assert max_exact_operand(1, torch.float16) == 2048
    # fp32 is accumulator-bound, and the answer is 4095 rather than sqrt(2**24) = 4096 because the
    # requirement is STRICT: at P = 4096 a single product already equals 2**24, which is the first
    # magnitude where the next integer up is not representable. One off the boundary, deliberately.
    assert max_exact_operand(1, torch.float32) == 4095


def test_integer_operands_are_integers_within_the_bound():
    """The generated values really are integers, and really are inside the requested range."""
    g = torch.Generator().manual_seed(0)
    t = integer_operands((64, 128), torch.float16, "cpu", generator=g, K=128)
    assert torch.equal(t.double(), t.double().round()), "values must be exact integers"
    assert float(t.abs().max()) <= max_exact_operand(128, torch.float16)
    assert float(t.abs().max()) > 0, "an all-zero operand would make every gate vacuous"


def test_integer_operands_refuses_a_bound_the_dtype_cannot_represent():
    """Asking for 4096 at bf16 raises: the operands would round and the argument would not hold."""
    with pytest.raises(ValueError, match=r"exceeds what .* represents exactly"):
        integer_operands((8, 8), torch.bfloat16, "cpu", bound=4096)


def test_integer_operands_refuses_a_K_that_leaves_no_room():
    """A K so large that P collapses to 0 raises instead of producing an all-zero operand."""
    with pytest.raises(ValueError, match=r"too large for an integer-exact test"):
        integer_operands((4, 4), torch.bfloat16, "cpu", K=2**26)


# ── the bitwise gate ──────────────────────────────────────────────────────────────────────────
def test_assert_bitwise_passes_on_identical_tensors():
    """The happy path, so a later failure means a real difference and not a broken gate."""
    a = torch.randn(4, 8)
    assert_bitwise(a, a.clone())


def test_assert_bitwise_catches_a_single_differing_element_and_names_it():
    """One element in 4096 must fail, and the message must give its coordinate and both values.

    This is the case the replaced scalar gate was worst at: one wrong element among thousands
    barely moves a max-over-max ratio.
    """
    a = torch.zeros(64, 64)
    b = a.clone()
    b[7, 13] = 1e-30  # utterly invisible to any relative-error summary
    with pytest.raises(AssertionError, match=r"1/4096 elements") as e:
        assert_bitwise(b, a, what="probe")
    assert "(7, 13)" in str(e.value), "the message must locate the offender"
    assert "probe" in str(e.value)


def test_assert_bitwise_reports_shape_and_dtype_mismatch_as_such():
    """A shape or dtype mismatch is diagnosed by name, not as a confusing element comparison."""
    with pytest.raises(AssertionError, match=r"shape mismatch"):
        assert_bitwise(torch.zeros(4, 4), torch.zeros(4, 5))
    with pytest.raises(AssertionError, match=r"dtype mismatch"):
        assert_bitwise(torch.zeros(4, 4, dtype=torch.float16), torch.zeros(4, 4))


def test_assert_bitwise_separates_negative_zero_from_positive_zero():
    """``-0.0 == 0.0`` is True in arithmetic but they are different bit patterns.

    A copy path that loses the sign of zero is a real defect -- it changes the sign of a subsequent
    division -- and a value comparison cannot see it. This gate is called *bitwise*, so it must.
    """
    a = torch.tensor([0.0, 1.0])
    b = torch.tensor([-0.0, 1.0])
    assert bool((a == b).all()), "precondition: these compare equal by value"
    with pytest.raises(AssertionError, match=r"differ bitwise"):
        assert_bitwise(b, a)


# ── the elementwise gate: the point of the whole module ───────────────────────────────────────
@numeric_exempt(
    "the OLD pooled gate IS this test's subject: it computes max|diff| / max|ref| and asserts "
    "it PASSES, which is the precondition that makes the element-wise failure beside it mean "
    "anything. Converting it deletes the comparison being made"
)
def test_elementwise_catches_what_a_tensor_wide_scalar_ratio_misses():
    """THE regression test for the gate this module replaces.

    Construct a reference with one huge element and one tiny one, then corrupt the tiny element
    *completely* -- it is off by 100% of its own value. The old scalar gate divides the max
    absolute error by the max absolute reference, so the corruption is diluted by the huge element
    and reads as a 1e-6 relative error: a pass. The per-element gate compares each element against
    its own bound and fails.
    """
    ref = torch.tensor([[1.0e6, 1.0e-3]], dtype=torch.float64)
    actual = torch.tensor([[1.0e6, 2.0e-3]], dtype=torch.float64)  # element 1 is 100% wrong
    bound = torch.tensor([[1.0e2, 1.0e-9]], dtype=torch.float64)

    old_gate = float((actual - ref).abs().max()) / float(ref.abs().max())
    assert old_gate < 0.02, f"precondition: the scalar gate passes this ({old_gate:.3g} < 0.02)"

    with pytest.raises(AssertionError, match=r"1/2 elements .* exceed their per-element bound"):
        assert_elementwise(actual, ref, bound)


def test_elementwise_returns_the_worst_ratio_when_it_passes():
    """A passing call reports how close it came, so drift is visible before it becomes a failure."""
    ref = torch.zeros(4, 4, dtype=torch.float64)
    actual = torch.full((4, 4), 0.5, dtype=torch.float64)
    worst = assert_elementwise(actual, ref, torch.ones(4, 4, dtype=torch.float64))
    assert worst == pytest.approx(0.5)


def test_elementwise_diagnoses_a_nan_as_a_nan():
    """A NaN is reported as non-finite, not as an unexplained bound violation.

    ``nan > bound`` is False, so a NaN can slip through a naive comparison entirely. The finiteness
    check runs first precisely so the message names the real problem.
    """
    ref = torch.zeros(2, 2, dtype=torch.float64)
    actual = torch.zeros(2, 2, dtype=torch.float64)
    actual[1, 1] = float("nan")
    with pytest.raises(AssertionError, match=r"non-finite element"):
        assert_elementwise(actual, ref, torch.ones(2, 2, dtype=torch.float64))


def test_elementwise_reports_the_worst_offenders_with_coordinates():
    """The failure message lists coordinates, values and ratios -- not a single summary number."""
    ref = torch.zeros(8, 8, dtype=torch.float64)
    actual = torch.zeros(8, 8, dtype=torch.float64)
    actual[2, 3] = 10.0
    actual[5, 1] = 5.0
    bound = torch.ones(8, 8, dtype=torch.float64)
    with pytest.raises(AssertionError) as e:
        assert_elementwise(actual, ref, bound)
    msg = str(e.value)
    assert "(2, 3)" in msg and "(5, 1)" in msg
    assert re.search(r"ratio=10\.0", msg), "the worst offender must be listed first with its ratio"


@pytest.mark.parametrize(
    "bound",
    [1.0, torch.tensor(1.0, dtype=torch.float64), torch.ones(1, 8, dtype=torch.float64)],
    ids=["python_float", "zero_dim_tensor", "broadcast_row"],
)
def test_a_scalar_or_broadcast_bound_still_reports_its_offenders(bound):
    """A uniform or broadcast bound must reach the REPORT, not just the comparison.

    The comparison broadcasts by itself, so a 0-d bound compares correctly and every passing test
    looks fine. Indexing it with a full coordinate for the message does not: it raised
    ``IndexError: too many indices for tensor of dimension 0`` -- on the FAILURE path only, which is
    the one run where the diagnostic was the whole point. All three legal spellings are swept
    because a fix for one of them is not a fix for the others: ``frac * ref.std()`` is 0-d, a plain
    ``atol`` is a Python float, and a per-column allowance is partially broadcast.
    """
    ref = torch.zeros(4, 8, dtype=torch.float64)
    actual = torch.zeros(4, 8, dtype=torch.float64)
    actual[1, 5] = 3.0
    with pytest.raises(AssertionError) as e:
        assert_elementwise(actual, ref, bound)
    msg = str(e.value)
    assert "(1, 5)" in msg, f"the offending coordinate must be named; got:\n{msg}"
    assert "bound=1" in msg, f"the bound must be reported per element; got:\n{msg}"


def test_a_comparison_larger_than_one_slice_reports_a_GLOBAL_coordinate():
    """A split comparison must be indistinguishable from a whole one, except in peak memory.

    ``assert_elementwise`` promotes BOTH operands to fp64, so the A2A-fused token ladder's
    ``(1, 2048, 2048, 512)`` output is 34 GB before a temporary -- the CHECK ran out of memory at a
    shape the kernel had just computed correctly, and outside the test's OOM-skip, so it read as a
    correctness failure. Splitting fixes that; reporting a slice-local index would replace it with
    a subtler wrong answer, a coordinate pointing at the wrong row.

    Shrinks ``_CHUNK_ELEMS`` rather than building a huge tensor, so what is under test is the
    arithmetic of the offset and it is checked in milliseconds.
    """
    # The offending element sits on the LONGEST axis (64 > 4), which is the one the split walks --
    # and deliberately not on axis 0 for the second shape below, because splitting the leading axis
    # is the obvious implementation and the one that silently did nothing on the shape that
    # motivated this: `(1, 2048, 2048, 512)` has a leading extent of 1.
    ref = torch.zeros(64, 4, dtype=torch.float64)
    actual = torch.zeros(64, 4, dtype=torch.float64)
    actual[50, 2] = 9.0
    with pytest.raises(AssertionError) as e:
        assert_elementwise(actual, ref, 1.0)
    whole = str(e.value)
    saved = numerics._CHUNK_ELEMS
    try:
        numerics._CHUNK_ELEMS = 8
        with pytest.raises(AssertionError) as e:
            assert_elementwise(actual, ref, 1.0)
        split = str(e.value)
    finally:
        numerics._CHUNK_ELEMS = saved
    assert "(50, 2)" in whole and "(50, 2)" in split, (
        f"the split comparison must name the same GLOBAL coordinate.\nwhole: {whole}\n"
        f"split: {split}"
    )
    assert "ratio=9.0" in split, f"the ratio must survive the split; got:\n{split}"

    # A leading extent of 1 -- the shape that motivated the fix. A leading-axis split yields one
    # slice of the whole tensor and saves nothing, so the axis chosen must be the longest one.
    ref = torch.zeros(1, 64, 4, dtype=torch.float64)
    actual = torch.zeros(1, 64, 4, dtype=torch.float64)
    actual[0, 50, 2] = 9.0
    saved = numerics._CHUNK_ELEMS
    try:
        numerics._CHUNK_ELEMS = 8
        with pytest.raises(AssertionError) as e:
            assert_elementwise(actual, ref, 1.0)
        thin = str(e.value)
    finally:
        numerics._CHUNK_ELEMS = saved
    assert "(0, 50, 2)" in thin, (
        f"a tensor whose LEADING extent is 1 must still be split, on its longest axis, and still "
        f"report a global coordinate; got:\n{thin}"
    )


# ── the analytic bound ────────────────────────────────────────────────────────────────────────
def test_the_bound_is_a_tensor_that_varies_per_element():
    """It must not collapse to one number -- that would be the scalar gate again, renamed.

    Element (0, j) sums |a*b| over rows whose magnitudes differ by construction, so the bounds must
    differ too.
    """
    A = torch.tensor([[1.0, 1.0], [1000.0, 1000.0]], dtype=torch.float16)
    B = torch.tensor([[1.0, 1.0]], dtype=torch.float16)
    bound = gemm_error_bound(A, B, torch.float16)
    assert bound.shape == (2, 1)
    assert float(bound[1, 0]) > float(bound[0, 0]) * 100, (
        "a row with 1000x the magnitude must get a proportionally larger bound"
    )


def test_the_bound_grows_with_K():
    """More accumulation steps means more permitted error; the bound must reflect that."""
    torch.manual_seed(0)
    small = gemm_error_bound(torch.ones(4, 8), torch.ones(4, 8), torch.float32)
    large = gemm_error_bound(torch.ones(4, 512), torch.ones(4, 512), torch.float32)
    assert float(large.mean()) > float(small.mean())


def test_the_bound_is_looser_for_a_narrower_output_type():
    """bf16 out gets a bigger store term than fp16 out, at identical operands and K."""
    A = torch.randn(8, 64)
    B = torch.randn(8, 64)
    b16 = float(gemm_error_bound(A, B, torch.bfloat16).mean())
    f16 = float(gemm_error_bound(A, B, torch.float16).mean())
    f32 = float(gemm_error_bound(A, B, torch.float32).mean())
    assert b16 > f16 > f32, f"expected bf16 > fp16 > fp32, got {b16:.3g} {f16:.3g} {f32:.3g}"


# ── the composed GEMM gates ───────────────────────────────────────────────────────────────────
def test_gemm_exact_passes_for_an_exactly_representable_product():
    """A CPU fp64 matmul of in-window integers is exact, so the gate must accept it."""
    g = torch.Generator().manual_seed(3)
    A = integer_operands((16, 32), torch.float16, "cpu", generator=g, K=32)
    B = integer_operands((8, 32), torch.float16, "cpu", generator=g, K=32)
    D = gemm_reference(A, B).to(torch.float32)
    assert_gemm_exact(D, A, B)


def test_gemm_exact_catches_a_one_ulp_error():
    """A single least-significant-bit difference fails -- there is no tolerance to hide in."""
    g = torch.Generator().manual_seed(4)
    A = integer_operands((8, 16), torch.float16, "cpu", generator=g, K=16)
    B = integer_operands((8, 16), torch.float16, "cpu", generator=g, K=16)
    D = gemm_reference(A, B).to(torch.float32)
    D[3, 5] += 1.0
    with pytest.raises(AssertionError, match=r"differ bitwise"):
        assert_gemm_exact(D, A, B)


def test_gemm_exact_refuses_non_integer_operands():
    """It verifies its own precondition instead of trusting the caller.

    A silently-violated precondition turns the strongest gate into a flaky one, which is worse
    than not having it: it would fail intermittently and be "fixed" by loosening something.
    """
    A, B = torch.randn(4, 8), torch.randn(4, 8)
    with pytest.raises(AssertionError, match=r"not integer-valued"):
        assert_gemm_exact(gemm_reference(A, B).float(), A, B)


def test_gemm_exact_refuses_operands_outside_the_exactness_window():
    """``K*P*Q`` at or past 2**24 is refused by name, pointing at max_exact_operand."""
    A = torch.full((4, 4096), 2048.0, dtype=torch.float16)
    B = torch.full((4, 4096), 2048.0, dtype=torch.float16)
    with pytest.raises(AssertionError, match=r"reaches fp32's exact-integer ceiling"):
        assert_gemm_exact(gemm_reference(A, B).float(), A, B)


def test_gemm_close_accepts_a_correctly_rounded_result():
    """An fp32 matmul of fp16 operands sits well inside the analytic bound."""
    torch.manual_seed(7)
    A = torch.randn(32, 128).half()
    B = torch.randn(16, 128).half()
    D = (A.float() @ B.float().T).half()
    worst = assert_gemm_close(D, A, B)
    assert worst < 1.0


def test_gemm_close_rejects_a_degenerate_all_zero_reference():
    """An all-zero reference would pass vacuously, so it is refused as a broken test instead."""
    A = torch.zeros(8, 16).half()
    B = torch.randn(8, 16).half()
    with pytest.raises(AssertionError, match=r"reference is all zeros"):
        assert_gemm_close(torch.zeros(8, 8).half(), A, B)


def test_gemm_close_catches_a_wrong_column():
    """Corrupting one output column fails, and the message locates it."""
    torch.manual_seed(8)
    A = torch.randn(32, 64).half()
    B = torch.randn(16, 64).half()
    D = (A.float() @ B.float().T).half()
    D[:, 5] = 0.0
    with pytest.raises(AssertionError, match=r"exceed their per-element bound"):
        assert_gemm_close(D, A, B)


# ── the "was it written at all" gate ──────────────────────────────────────────────────────────
def test_assert_written_accepts_a_kernel_that_fills_everything():
    """A complete write passes and hands back the output for the caller's value checks."""
    out = assert_written(lambda d: d.fill_(3.0), (4, 8), torch.float32, "cpu")
    assert out.shape == (4, 8) and float(out.min()) == 3.0


def test_assert_written_catches_a_skipped_tail():
    """A store that stops short leaves the pre-fill behind, and the two runs disagree there.

    This is the failure a value comparison misses whenever the reference happens to be near zero
    in the skipped region -- a partial trailing tile, which is exactly where predication bugs live.
    """

    def partial(d):
        d[:, :6] = 3.0  # columns 6 and 7 never written

    with pytest.raises(AssertionError, match=r"never written"):
        assert_written(partial, (4, 8), torch.float32, "cpu")


# ── the gated bound, which carries error THROUGH a nonlinearity ────────────────────────────────
def _gated_reference(A, Wg, Wp):
    """The exact fp64 dual-gated result, computed without any of the kernel's shortcuts."""
    a = A.double()
    return torch.sigmoid(a @ Wg.double().T) * (a @ Wp.double().T)


def test_the_gated_bound_is_a_tensor_that_varies_per_element():
    """Same reason the GEMM bound is: an element whose up-projection is small deserves a tight
    bound, and a scalar would hand it the whole tensor's slack."""
    torch.manual_seed(0)
    A, Wg = torch.randn(8, 32), torch.randn(6, 32)
    Wp = torch.randn(6, 32)
    Wp[0] *= 1e-3  # one nearly-zero output column
    bound = gated_error_bound(A, Wg, Wp, _gated_reference(A, Wg, Wp), torch.bfloat16)
    assert bound.shape == (8, 6)
    assert bound[:, 0].max() < bound[:, 1:].min(), (
        "the column with a tiny up projection should get a tighter bound than the others"
    )


def test_the_gated_bound_admits_a_faithful_computation():
    """An fp32 emulation of the kernel's own arithmetic must sit inside the bound everywhere."""
    torch.manual_seed(0)
    A, Wg, Wp = torch.randn(16, 64), torch.randn(12, 64), torch.randn(12, 64)
    ref = _gated_reference(A, Wg, Wp)
    got = (torch.sigmoid(A @ Wg.T) * (A @ Wp.T)).to(torch.bfloat16)
    assert_elementwise(got, ref, gated_error_bound(A, Wg, Wp, ref, torch.bfloat16))


def test_the_gated_bound_rejects_a_swapped_gate_and_up():
    """The one mistake the fold can make silently: gating the up value by the up value.

    Feeding the gate a deliberately wrong tensor is what says the bound has teeth -- a bound wide
    enough to admit this would admit a mis-paired register too.
    """
    torch.manual_seed(0)
    A, Wg, Wp = torch.randn(16, 64), torch.randn(12, 64), torch.randn(12, 64)
    ref = _gated_reference(A, Wg, Wp)
    swapped = (torch.sigmoid(A @ Wp.T) * (A @ Wg.T)).to(torch.bfloat16)
    with pytest.raises(AssertionError):
        assert_elementwise(swapped, ref, gated_error_bound(A, Wg, Wp, ref, torch.bfloat16))


def test_the_hardware_sigmoid_is_the_largest_term_in_the_gated_bound():
    """Pins the docstring's claim about WHICH term dominates, by recomputing the terms separately.

    The three contributions before the store are the sigmoid's own ``2**-12`` absolute error, the
    gate pre-activation's accumulation error carried through the sigmoid's slope, and the up
    pre-activation's accumulation error. At a realistic K the first is the largest -- which is why
    the gated bound is not the GEMM bound with a constant on it, and why an ULP-scale tolerance
    fails here.

    Note this does NOT say the gated bound exceeds `gemm_error_bound` for the same operands: that
    bound's store term is taken against the sum of ABSOLUTE products, which is much larger than the
    gated reference, so the two are not comparable term by term.
    """
    torch.manual_seed(0)
    A, Wg, Wp = torch.randn(8, 128), torch.randn(8, 128), torch.randn(8, 128)
    K = A.shape[-1]
    gamma = (K * 2.0**-24) / (1.0 - K * 2.0**-24)
    a64 = A.double().abs()
    abs_up = a64 @ Wp.double().abs().T
    e_gate = gamma * (a64 @ Wg.double().abs().T)
    e_up = gamma * abs_up
    sigmoid_term = 2.0**-12 * abs_up
    gate_term = 0.25 * e_gate * abs_up
    assert sigmoid_term.median() > gate_term.median()
    assert sigmoid_term.median() > e_up.median()


def test_the_gated_bound_refuses_a_K_the_gamma_formula_cannot_describe():
    """Same guard the GEMM bound carries; it needs K > 2**24, far past anything real."""
    A = torch.zeros(1, 2**25)
    W = torch.zeros(1, 2**25)
    with pytest.raises(ValueError, match=r"too large for the gamma bound"):
        gated_error_bound(A, W, W, torch.zeros(1, 1), torch.bfloat16)


# ── the algebraic-fold bound ──────────────────────────────────────────────────────────────────
# Everything here runs on the CPU deliberately. The simulation below multiplies in fp32 to model
# the kernel's accumulator, and a CUDA fp32 matmul may silently be TF32 -- a 10-bit mantissa, which
# would make the simulated error an order of magnitude larger than the arithmetic being modelled and
# turn a passing bound into a failing one for a reason that has nothing to do with the bound.


def _simulate_alg_fold(x, w, b, Wg, Wp, eps, bg=None, bp=None, out_dtype=torch.bfloat16):
    """Reproduce, in torch, the arithmetic ``alg_fold`` performs -- at the widths it performs it.

    This is the subject :func:`alg_fold_error_bound` claims to bound, so it exists to let the claim
    be checked without a GPU or a kernel. It is NOT a reference implementation: it is deliberately
    the *inaccurate* computation, written to round where the kernel rounds.

    Modelled faithfully: the fold ``diag(w) @ B`` rounded to the weight dtype and materialized; the
    fp32 accumulation of ``x @ Bw``; the fp32 ``c`` reduction over the ROUNDED fold; the ``d``
    reduction over the unrounded weight; the ``E[x^2] - mu^2`` statistics in fp32; the rank-one
    correction in fp32; the store rounded to `out_dtype`.

    NOT modelled: the kernel's tanh-approximation sigmoid (``2**-12`` absolute), its reduction
    ORDER, and its fast-math ``rsqrt``. The bound allows for all three, so a result from this
    simulation sits strictly inside the bound with room to spare -- which is the correct direction,
    but means these tests do not exercise those three terms.

    Args:
        x: ``(M, K)`` activation at the operand dtype. Must be on the CPU (see the module note
            above); a CUDA tensor would make the fp32 matmul below possibly TF32.
        w: ``(K,)`` fp32 LayerNorm gain.
        b: ``(K,)`` fp32 LayerNorm bias, or None.
        Wg: ``(N, K)`` gate weight at the operand dtype.
        Wp: ``(N, K)`` up weight, same shape and dtype as `Wg`.
        eps: LayerNorm variance floor.
        bg: ``(N,)`` gate projection bias, or None.
        bp: ``(N,)`` up projection bias, or None.
        out_dtype: The dtype the gated result is stored as.

    Returns:
        ``(M, N)`` tensor in `out_dtype` -- what a correct ``alg_fold`` kernel should produce, up to
        the three unmodelled terms above.
    """
    K = x.shape[-1]
    xf = x.float()
    mu = xf.sum(-1, keepdim=True) / K
    var = (xf * xf).sum(-1, keepdim=True) / K - mu * mu
    r = torch.rsqrt(var + eps)
    s = r * mu

    def half(W, proj_bias):
        exact_fold = w.float().unsqueeze(0) * W.float()  # (N, K), fp32, before the store rounds it
        bw = exact_fold.to(W.dtype)  # what the precompute kernel materializes
        acc = xf @ bw.float().transpose(-1, -2)  # the MMA reads the ROUNDED fold ...
        c = exact_fold.sum(-1)  # ... while colsum reduces the UNROUNDED one. Not a typo:
        # `_fold_precompute_kernel` explains why that asymmetry is the convention in force.
        pre = r * acc - s * c.unsqueeze(0)
        if b is not None:
            pre = pre + (b.float().unsqueeze(0) * W.float()).sum(-1).unsqueeze(0)
        if proj_bias is not None:
            pre = pre + proj_bias.float().unsqueeze(0)
        return pre

    return (torch.sigmoid(half(Wg, bg)) * half(Wp, bp)).to(out_dtype)


def _simulate_alg_fold_gate3(x, w, b, W3, eps, b3=None, out_dtype=torch.bfloat16):
    """Reproduce `alg_fold`'s OUTPUT-GATE arithmetic: one projection, one sigmoid, no pairing.

    The gate reads the same ``LN(x)`` as the dual and is repaired by the same rank-one identity,
    so this is `_simulate_alg_fold` with a single weight and the glu pairing removed. It is the
    subject :func:`alg_fold_sigmoid_error_bound` claims to bound.

    Args:
        x: ``(M, K)`` activation at the operand dtype, on the CPU (see the module note above).
        w: ``(K,)`` fp32 LayerNorm gain.
        b: ``(K,)`` fp32 LayerNorm bias, or None.
        W3: ``(N3, K)`` gate weight at the operand dtype.
        eps: LayerNorm variance floor. Must be the value the bound is given.
        b3: ``(N3,)`` gate bias, or None.
        out_dtype: The dtype the gate output is stored as.

    Returns:
        ``(M, N3)`` tensor in `out_dtype`.
    """
    return torch.sigmoid(_alg_fold_gate3_preact(x, w, b, W3, eps, b3)).to(out_dtype)


def _alg_fold_gate3_preact(x, w, b, W3, eps, b3=None):
    """The gate's pre-activation under `alg_fold`, at the widths the kernel forms it.

    Args:
        x, w, b, W3, eps, b3: As :func:`_simulate_alg_fold_gate3`.

    Returns:
        ``(M, N3)`` fp32 pre-activation.
    """
    K = x.shape[-1]
    xf = x.float()
    mu = xf.sum(-1, keepdim=True) / K
    var = (xf * xf).sum(-1, keepdim=True) / K - mu * mu
    r = torch.rsqrt(var + eps)
    s = r * mu
    exact_fold = w.float().unsqueeze(0) * W3.float()
    stored_fold = exact_fold.to(W3.dtype)
    acc = xf @ stored_fold.float().transpose(-1, -2)
    # The gate reduces the ROUNDED fold, the OPPOSITE of the dual's convention above. Not a typo:
    # the upstream uses both, in one file, and `build_folded_gate3_operands` records why the
    # asymmetry is reproduced rather than resolved.
    pre = r * acc - s * stored_fold.float().sum(-1).unsqueeze(0)
    if b is not None:
        pre = pre + (b.float().unsqueeze(0) * W3.float()).sum(-1).unsqueeze(0)
    if b3 is not None:
        pre = pre + b3.float().unsqueeze(0)
    return pre


def _alg_fold_case(M=64, N=32, K=128, dtype=torch.float16, mean=0.0, seed=0, bias=True):
    """Build one well-conditioned ``alg_fold`` case and its exact fp64 reference.

    Args:
        M, N, K: Problem dims. `K` is the LayerNorm width AND the GEMM contraction, as in the kernel.
        dtype: Operand dtype for `x`, `Wg`, `Wp`. fp16 is the default because its 11-bit mantissa
            gives the fold rounding room to be visible without dominating.
        mean: Per-row offset added to `x`. 0 keeps the rows centred; a large value drives the
            ``r*(x@Bw)`` vs ``s*c`` cancellation the bound's docstring describes, which is what
            :func:`test_the_alg_fold_bound_widens_where_the_fusion_is_ill_conditioned` uses.
        seed: RNG seed, so a failure is reproducible.
        bias: Whether to build a LayerNorm bias and projection biases.

    Returns:
        A dict with ``x, w, b, Wg, Wp, bg, bp, eps, reference`` -- `reference` being the exact fp64
        gated output, biases included, which is what the bound must be taken against.
    """
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(M, K, generator=g) + mean).to(dtype)
    w = torch.randn(K, generator=g)
    b = torch.randn(K, generator=g) if bias else None
    Wg = (torch.randn(N, K, generator=g) / K**0.5).to(dtype)
    Wp = (torch.randn(N, K, generator=g) / K**0.5).to(dtype)
    bg = torch.randn(N, generator=g).to(dtype) if bias else None
    bp = torch.randn(N, generator=g).to(dtype) if bias else None
    eps = 1e-5
    an = torch.nn.functional.layer_norm(
        x.double(), (K,), w.double(), None if b is None else b.double(), eps
    )
    gate = an @ Wg.double().transpose(-1, -2)
    up = an @ Wp.double().transpose(-1, -2)
    if bg is not None:
        gate = gate + bg.double()
        up = up + bp.double()
    return dict(
        x=x, w=w, b=b, Wg=Wg, Wp=Wp, bg=bg, bp=bp, eps=eps, reference=torch.sigmoid(gate) * up
    )


def _bound_for(case, out_dtype=torch.bfloat16):
    """Call :func:`alg_fold_error_bound` for a case dict from :func:`_alg_fold_case`.

    Args:
        case: The dict :func:`_alg_fold_case` returns.
        out_dtype: Store dtype to bound for.

    Returns:
        The fp64 per-element bound.
    """
    return alg_fold_error_bound(
        case["x"],
        case["w"],
        case["b"],
        case["Wg"],
        case["Wp"],
        case["reference"],
        case["eps"],
        out_dtype,
        bg=case["bg"],
        bp=case["bp"],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
def test_the_alg_fold_bound_holds_for_the_arithmetic_it_models(dtype, bias):
    """The simulated kernel's every element sits inside the bound.

    This is the whole claim. If it fails, either the derivation is wrong or the simulation rounds
    somewhere the derivation does not know about -- and the two are worth distinguishing before
    anything is blamed on a kernel.
    """
    case = _alg_fold_case(dtype=dtype, bias=bias)
    out = _simulate_alg_fold(
        case["x"],
        case["w"],
        case["b"],
        case["Wg"],
        case["Wp"],
        case["eps"],
        case["bg"],
        case["bp"],
        torch.bfloat16,
    )
    err = (out.double() - case["reference"]).abs()
    bound = _bound_for(case)
    worst = (err / bound).max().item()
    assert worst <= 1.0, f"worst element exceeded its bound by {worst:.3f}x"


def test_the_alg_fold_bound_is_not_vacuous():
    """A result twice as wrong as the bound allows is caught.

    A bound that always passes is worse than none, because it reads as evidence. This perturbs one
    element past its own allowance and checks the comparison notices.
    """
    case = _alg_fold_case()
    bound = _bound_for(case)
    out = _simulate_alg_fold(
        case["x"],
        case["w"],
        case["b"],
        case["Wg"],
        case["Wp"],
        case["eps"],
        case["bg"],
        case["bp"],
        torch.bfloat16,
    ).double()
    out[3, 5] = out[3, 5] + 2.0 * bound[3, 5]
    err = (out - case["reference"]).abs()
    assert (err > bound).sum().item() == 1
    assert bool((err > bound)[3, 5])


def test_the_alg_fold_bound_varies_per_element():
    """A tensor, not a scalar broadcast -- the reason every bound in this module is a tensor."""
    bound = _bound_for(_alg_fold_case())
    assert bound.shape == (64, 32)
    assert bound.min().item() < bound.max().item()


def test_the_alg_fold_bound_grows_with_K():
    """More contraction terms, more accumulated error -- at fixed per-element magnitude."""
    small = _bound_for(_alg_fold_case(K=64)).median().item()
    large = _bound_for(_alg_fold_case(K=512)).median().item()
    assert large > small


def test_the_alg_fold_bound_is_looser_for_a_narrower_store():
    """The store term tracks the output dtype, so bf16 gets more room than fp32."""
    case = _alg_fold_case()
    assert _bound_for(case, torch.bfloat16).median() > _bound_for(case, torch.float32).median()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_the_two_fusions_cross_over_rather_than_one_dominating(dtype):
    """The two variants' accuracy ordering REVERSES with the row offset, and both directions hold.

    This is the claim the bound's docstring makes and the reason it exists as a separate function.
    On a centred row ``alg_fold`` is the more accurate of the two -- it never rounds a normalized
    activation to 16 bits, which is the prologue variant's dominant term. On a strongly offset row it
    is the less accurate -- ``r*(x@Bw) - s*c`` becomes a difference of two large nearly-equal
    quantities, which the prologue variant has no analogue of.

    Asserting only the second half (the version this test replaced) would leave the first half
    untested, and the first half is the counter-intuitive one: a reader who assumes the algebraic
    fold is simply "the less accurate variant" would be wrong at every offset the workflow's
    activations are likely to have, and would pick a variant on that belief. Both bounds are taken on
    the IDENTICAL input, so the ratio isolates the structural difference and nothing else.
    """
    ratios = []
    for mean in (0.0, 100.0):
        case = _alg_fold_case(dtype=dtype, mean=mean)
        alg = _bound_for(case).median().item()
        prolog = (
            fused_ln_gated_error_bound(
                case["x"],
                case["w"],
                case["b"],
                case["Wg"],
                case["Wp"],
                case["reference"],
                case["eps"],
                torch.bfloat16,
            )
            .median()
            .item()
        )
        ratios.append(alg / prolog)
    assert ratios[0] < 1.0, f"alg_fold should be TIGHTER on a centred row, got {ratios[0]:.3f}x"
    assert ratios[1] > 1.0, f"alg_fold should be LOOSER on an offset row, got {ratios[1]:.3f}x"
    assert ratios[1] / ratios[0] > 10.0, f"the crossover is barely there: {ratios}"


def test_the_alg_fold_bound_refuses_mismatched_weight_dtypes():
    """The two halves live in ONE interleaved tensor, so they cannot be rounded at two widths."""
    case = _alg_fold_case()
    with pytest.raises(ValueError, match="one width"):
        alg_fold_error_bound(
            case["x"],
            case["w"],
            case["b"],
            case["Wg"],
            case["Wp"].bfloat16(),
            case["reference"],
            case["eps"],
            torch.bfloat16,
        )


def test_the_alg_fold_bound_refuses_a_K_the_gamma_formula_cannot_describe():
    """``K*u >= 1`` makes ``gamma_K`` negative; refuse rather than return a nonsense bound."""
    case = _alg_fold_case(K=128)
    x = case["x"][:, :1].expand(-1, 2**25).contiguous()
    with pytest.raises(ValueError, match="too large for the gamma bound"):
        alg_fold_error_bound(
            x,
            torch.ones(2**25),
            None,
            torch.ones(2, 2**25, dtype=torch.float16),
            torch.ones(2, 2**25, dtype=torch.float16),
            torch.zeros(x.shape[0], 2, dtype=torch.float64),
            1e-5,
            torch.bfloat16,
        )


# ── the algebraic fold's OUTPUT-GATE bound ────────────────────────────────────────────────────


def test_the_alg_fold_sigmoid_bound_admits_the_arithmetic_it_describes():
    """The simulated gate output sits inside `alg_fold_sigmoid_error_bound`, at every element."""
    c = _alg_fold_case(bias=True)
    ref3 = torch.sigmoid(
        torch.nn.functional.layer_norm(
            c["x"].double(), (c["x"].shape[-1],), c["w"].double(), c["b"].double(), c["eps"]
        )
        @ c["Wg"].double().transpose(-1, -2)
        + c["bg"].double()
    )
    got = _simulate_alg_fold_gate3(c["x"], c["w"], c["b"], c["Wg"], c["eps"], b3=c["bg"])
    bound = alg_fold_sigmoid_error_bound(
        c["x"], c["w"], c["b"], c["Wg"], ref3, c["eps"], torch.bfloat16, b3=c["bg"]
    )
    assert_elementwise(got, ref3, bound, what="alg_fold gate3")


def test_the_alg_fold_sigmoid_bound_is_small_against_the_range_it_gates():
    """The bound must be a small fraction of ``(0, 1)``, or it cannot fail on a wrong gate.

    This is the property that matters, and it is NOT implied by the bound being correct: a sound
    bound wide enough to span the output range accepts everything. The gate output lives in
    ``(0, 1)``, so the allowance is asserted against 1, not against the dual bound -- the two
    differ by only ~5x here (0.21 measured), because the dual's extra factor is the up-projection
    MAGNITUDE, which for these operands is order 1 rather than order 1000.

    The defect this bound exists to catch was a gate wrong by ~0.9 absolute. Any allowance below a
    few percent catches it by orders of magnitude; the assertion is set at 5% so that a future
    change loosening the bound tenfold still fails here rather than quietly passing.
    """
    c = _alg_fold_case(bias=True)
    ref3 = torch.sigmoid(
        torch.nn.functional.layer_norm(
            c["x"].double(), (c["x"].shape[-1],), c["w"].double(), c["b"].double(), c["eps"]
        )
        @ c["Wg"].double().transpose(-1, -2)
        + c["bg"].double()
    )
    gate = alg_fold_sigmoid_error_bound(
        c["x"], c["w"], c["b"], c["Wg"], ref3, c["eps"], torch.bfloat16, b3=c["bg"]
    )
    assert float(gate.max()) < 0.05, (
        f"the gate bound reaches {float(gate.max()):.3e} of an output confined to (0, 1); check "
        "that the sigmoid SLOPE factor is applied rather than a magnitude"
    )


def _simulate_alg_fold_xgate(
    x, x_gate, w, b, Wg, Wp, eps, bg=None, bp=None, out_dtype=torch.bfloat16
):
    """Reproduce the TWO-A kernel's arithmetic: a plain gate GEMM, a folded value projection.

    The subject :func:`alg_fold_xgate_error_bound` claims to bound. It differs from
    :func:`_simulate_alg_fold` in the two ways that matter and in no others:

    * the GATE arm is a plain fp32 accumulation of ``x_gate @ Wg`` -- no fold, no ``c``, no ``d``,
      no statistics, because ``x_gate`` arrived normalized;
    * the VALUE arm's ``c`` reduces the ROUNDED fold, the convention
      `build_folded_gate3_operands` uses, where the dual's reduces the unrounded one. Simulating
      the dual's convention here would put a term in the SIMULATION that the kernel does not
      incur, and the test would then be checking a bound against the wrong arithmetic.

    Args:
        x: ``(M, K)`` value activation at the operand dtype, on the CPU (see the module note).
        x_gate: ``(M, K)`` gate activation, same shape and dtype, PRE-NORMALIZED.
        w: ``(K,)`` fp32 LayerNorm gain. Applies to the value arm only.
        b: ``(K,)`` fp32 LayerNorm bias, or None.
        Wg: ``(N, K)`` gate weight at the operand dtype, RAW.
        Wp: ``(N, K)`` value weight, same shape and dtype.
        eps: LayerNorm variance floor.
        bg: ``(N,)`` gate projection bias, or None.
        bp: ``(N,)`` value projection bias, or None.
        out_dtype: The dtype the combined result is stored as.

    Returns:
        ``(M, N)`` tensor in `out_dtype`.
    """
    K = x.shape[-1]
    xf = x.float()
    mu = xf.sum(-1, keepdim=True) / K
    var = (xf * xf).sum(-1, keepdim=True) / K - mu * mu
    r = torch.rsqrt(var + eps)
    s = r * mu
    # GATE: a plain 16-bit GEMM. Nothing is folded in and nothing is repaired.
    gate = x_gate.float() @ Wg.float().transpose(-1, -2)
    if bg is not None:
        gate = gate + bg.float().unsqueeze(0)
    # VALUE: the fold, at the ROUNDED-colsum convention this path folds with.
    exact_fold = w.float().unsqueeze(0) * Wp.float()
    bw = exact_fold.to(Wp.dtype)
    acc = xf @ bw.float().transpose(-1, -2)
    c = bw.float().sum(-1)
    up = r * acc - s * c.unsqueeze(0)
    if b is not None:
        up = up + (b.float().unsqueeze(0) * Wp.float()).sum(-1).unsqueeze(0)
    if bp is not None:
        up = up + bp.float().unsqueeze(0)
    return (torch.sigmoid(gate) * up).to(out_dtype)


def _xgate_case(M=64, N=48, K=128, dtype=torch.bfloat16, seed=5, bias=True):
    """Build one two-A case plus its exact fp64 reference.

    ``x_gate`` is drawn INDEPENDENTLY of ``x``: deriving it from ``x`` would make a bound that
    confused the two arms look correct.

    Args:
        M: Rows. N: Output width. K: Contraction extent. Unconstrained; K must satisfy the gamma
            formula.
        dtype: Operand dtype, 16-bit.
        seed: RNG seed.
        bias: Whether to build a LayerNorm bias and both projection biases.

    Returns:
        A dict with ``x, x_gate, w, b, Wg, Wp, bg, bp, eps, reference``.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, K, generator=g).to(dtype)
    x_gate = torch.randn(M, K, generator=g).to(dtype)
    w = torch.randn(K, generator=g)
    b = torch.randn(K, generator=g) if bias else None
    Wg = (torch.randn(N, K, generator=g) / K**0.5).to(dtype)
    Wp = (torch.randn(N, K, generator=g) / K**0.5).to(dtype)
    bg = torch.randn(N, generator=g).to(dtype) if bias else None
    bp = torch.randn(N, generator=g).to(dtype) if bias else None
    eps = 1e-5
    an = torch.nn.functional.layer_norm(
        x.double(), (K,), w.double(), None if b is None else b.double(), eps
    )
    gate = x_gate.double() @ Wg.double().transpose(-1, -2)
    up = an @ Wp.double().transpose(-1, -2)
    if bias:
        gate = gate + bg.double()
        up = up + bp.double()
    return dict(
        x=x,
        x_gate=x_gate,
        w=w,
        b=b,
        Wg=Wg,
        Wp=Wp,
        bg=bg,
        bp=bp,
        eps=eps,
        reference=torch.sigmoid(gate) * up,
    )


def _xgate_bound_for(case, out_dtype=torch.bfloat16):
    """Call :func:`alg_fold_xgate_error_bound` for a case dict from :func:`_xgate_case`.

    Args:
        case: The dict :func:`_xgate_case` returns.
        out_dtype: Store dtype to bound for.

    Returns:
        The fp64 per-element bound.
    """
    return alg_fold_xgate_error_bound(
        case["x"],
        case["x_gate"],
        case["w"],
        case["b"],
        case["Wg"],
        case["Wp"],
        case["reference"],
        case["eps"],
        out_dtype,
        bg=case["bg"],
        bp=case["bp"],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
@numeric_exempt(
    "asserts max(|err| / per-element bound) <= 1, which is assert_elementwise's own logic "
    "inlined and is element-wise-EQUIVALENT -- the max is over per-element RATIOS, not over a "
    "magnitude divided by a pooled one, so it cannot shadow an element"
)
def test_the_two_a_bound_holds_for_the_arithmetic_it_models(dtype, bias):
    """The simulated two-A kernel's every element sits inside the bound. The whole claim."""
    c = _xgate_case(dtype=dtype, bias=bias)
    out = _simulate_alg_fold_xgate(
        c["x"],
        c["x_gate"],
        c["w"],
        c["b"],
        c["Wg"],
        c["Wp"],
        c["eps"],
        c["bg"],
        c["bp"],
        torch.bfloat16,
    )
    worst = ((out.double() - c["reference"]).abs() / _xgate_bound_for(c)).max().item()
    assert worst <= 1.0, f"worst element exceeded its bound by {worst:.3f}x"


def test_the_two_a_bound_is_not_vacuous():
    """A result past its own allowance is caught. A bound that always passes reads as evidence."""
    c = _xgate_case()
    bound = _xgate_bound_for(c)
    out = _simulate_alg_fold_xgate(
        c["x"],
        c["x_gate"],
        c["w"],
        c["b"],
        c["Wg"],
        c["Wp"],
        c["eps"],
        c["bg"],
        c["bp"],
    ).double()
    out[0, 0] = out[0, 0] + 2.0 * bound[0, 0]
    with pytest.raises(AssertionError):
        assert_elementwise(out, c["reference"], bound, what="perturbed two-A")


def test_reusing_the_all_folded_bound_here_would_accept_a_wrong_result():
    """**The test that justifies a third bound rather than reuse.** Stated as a defect it misses.

    `alg_fold_error_bound` models BOTH arms as folded-and-repaired. On this path the gate arm is a
    plain GEMM of an already-normalized activation and incurs none of the fold's terms -- no weight
    rounding, no rank-one cancellation -- so that bound hands it an allowance for errors it cannot
    make. This constructs an error in exactly that gap and shows the two bounds DISAGREE about it:
    the two-A bound rejects it, the all-folded one accepts it.

    A "the new bound is smaller everywhere" assertion was tried first and is FALSE, which is worth
    recording: the two also differ on the VALUE arm, where this path's ``c`` reduces the rounded
    fold and the dual's reduces the unrounded one, so neither dominates element-wise. Only the
    soundness consequence is a real property, and it is the one that matters.
    """
    c = _xgate_case()
    mine = _xgate_bound_for(c)
    as_folded = alg_fold_error_bound(
        c["x"],
        c["w"],
        c["b"],
        c["Wg"],
        c["Wp"],
        c["reference"],
        c["eps"],
        torch.bfloat16,
        bg=c["bg"],
        bp=c["bp"],
    )
    gap = as_folded - mine
    assert float(gap.max()) > 0, (
        "the all-folded bound is nowhere looser than this one, so the gate arm is not being "
        "modelled as a plain GEMM -- check that it does not go through `_alg_fold_preact_error`"
    )
    i, j = [int(v) for v in torch.nonzero(gap == gap.max())[0]]
    out = c["reference"].clone()
    # An error strictly between the two allowances: real by this path's arithmetic, invisible to
    # the bound that thinks the gate arm was folded.
    out[i, j] = out[i, j] + 0.5 * (mine[i, j] + as_folded[i, j])
    with pytest.raises(AssertionError):
        assert_elementwise(out, c["reference"], mine, what="two-A bound")
    assert_elementwise(out, c["reference"], as_folded, what="all-folded bound")


def test_the_two_a_bound_refuses_activations_that_disagree():
    """``x`` and ``x_gate`` must match in shape and dtype: the kernel contracts both against one K."""
    c = _xgate_case()
    with pytest.raises(ValueError, match=r"x_gate.*must\s+match"):
        alg_fold_xgate_error_bound(
            c["x"],
            c["x_gate"].to(torch.float16),
            c["w"],
            c["b"],
            c["Wg"],
            c["Wp"],
            c["reference"],
            c["eps"],
            torch.bfloat16,
        )


def test_the_two_a_bound_refuses_mismatched_weight_dtypes():
    """Both weights are operands of one kernel at one width; a mismatch is not representable."""
    c = _xgate_case()
    with pytest.raises(ValueError, match=r"must match"):
        alg_fold_xgate_error_bound(
            c["x"],
            c["x_gate"],
            c["w"],
            c["b"],
            c["Wg"].to(torch.float16),
            c["Wp"],
            c["reference"],
            c["eps"],
            torch.bfloat16,
        )


def test_the_two_a_bound_refuses_a_K_the_gamma_formula_cannot_describe():
    """Same guard every bound here carries."""
    z = torch.zeros(2, 2**25, dtype=torch.float16)
    with pytest.raises(ValueError, match=r"too large for the gamma bound"):
        alg_fold_xgate_error_bound(
            z,
            z,
            torch.zeros(2**25),
            None,
            torch.ones(2, 2**25, dtype=torch.float16),
            torch.ones(2, 2**25, dtype=torch.float16),
            torch.zeros(2, 2, dtype=torch.float64),
            1e-5,
            torch.bfloat16,
        )


def test_the_alg_fold_sigmoid_bound_refuses_a_K_the_gamma_formula_cannot_describe():
    """Same guard every bound here carries, reached through the shared terms helper."""
    x = torch.zeros(2, 2**25, dtype=torch.float16)
    with pytest.raises(ValueError, match=r"too large for the gamma bound"):
        alg_fold_sigmoid_error_bound(
            x,
            torch.zeros(2**25),
            None,
            torch.ones(2, 2**25, dtype=torch.float16),
            torch.zeros(2, 2, dtype=torch.float64),
            1e-5,
            torch.bfloat16,
        )


# ── the row-sum reduction bound ────────────────────────────────────────────────────────────────
#
# Added with the widening of the pooled detector, which found four tests gating a row-sum kernel on
# `((got - ref).norm() / ref.norm()).item() < BAR` -- a pooled L2 over every row, so one badly-wrong
# row among sixty-four cannot move it. Converting them needed a per-row bound, and a derived one
# rather than a hand-written atol/rtol.
#
# The pair below is the non-vacuity argument, in both directions: the bound must be large enough
# that a CORRECT fp32 reduction never trips it, and small enough that a REALISTIC defect does.


@pytest.mark.parametrize("K", [1024, 4096, 16384, 32768])
def test_reduction_bound_admits_a_correct_fp32_reduction(K):
    """torch's own fp32 row sum is inside the bound at every declared K, with orders to spare.

    The acceptance half. If this failed, every converted test would go red on correct data and the
    bound would be useless in the direction that matters most -- a gate that cries wolf gets
    loosened until it stops gating.
    """
    torch.manual_seed(0)
    x = torch.randn(64, K, dtype=torch.float32)
    assert_elementwise(
        x.float().sum(-1).double(),
        reduction_reference(x),
        reduction_error_bound(x, torch.float32),
        what=f"fp32 row sum at K={K}",
    )


@pytest.mark.parametrize("K", [1024, 4096, 16384, 32768])
def test_reduction_bound_is_not_vacuous_against_a_dropped_lane(K):
    """A silently dropped 1-lane-in-32 partial is caught on the great majority of rows.

    The refusal half, and the one that earns the bound its place: "derived" does not by itself mean
    "useful". It also pins the DEFAULT, because the obvious choice is wrong here -- at
    ``parallel_lanes=1`` (the sequential form ``gemm_error_bound`` uses, and calls safely
    pessimistic) detection collapses from 99.2% at K=1024 to **10.5%** at K=32768, because a
    zero-mean sum cancels to ``~sqrt(K)`` while ``sum|x|`` grows as ``K``. The measured comparison
    is in `reduction_error_bound`'s docstring.

    **A RATE, not ``.all()``, and that distinction was itself a measurement error worth keeping.**
    An earlier version of this test asserted every row detects it, justified by comparing the MAX
    bound to the MAX defect across rows -- two different rows, so the comparison meant nothing. Per
    row, a dropped lane's own terms can cancel to near zero; that is a property of the defect, not
    a weakness of the bound, and no bound can see a defect that contributes nothing.
    """
    torch.manual_seed(0)
    x = torch.randn(4096, K, dtype=torch.float32)
    damaged = x.clone()
    damaged[:, : K // 32] = 0.0  # one lane of 32 never contributed
    bound = reduction_error_bound(x, torch.float32)
    err = (damaged.double().sum(-1) - reduction_reference(x)).abs()
    detected = (err > bound).double().mean().item()
    assert detected >= 0.95, (
        f"K={K}: a dropped 1-in-32 lane is caught on only {100 * detected:.1f}% of rows, so the "
        f"bound is going vacuous for the defect class it exists for. Fix the DEPTH "
        f"(parallel_lanes), not the bar."
    )


def test_reduction_bound_refuses_a_K_the_gamma_formula_cannot_express():
    """``K * u >= 1`` makes ``gamma`` meaningless, so it raises rather than returning a negative."""
    with pytest.raises(ValueError, match=r"too large for the gamma bound"):
        reduction_error_bound(torch.zeros(2, 2**25, dtype=torch.float32), torch.float32)


def test_reduction_bound_is_tighter_for_a_row_that_cancels():
    """The bound follows ``sum|x|``, so a cancelling row is judged on ITS terms, not the tensor's.

    This is the property a pooled ratio cannot have and the reason the bound is a tensor. Row 0 is
    tiny in magnitude and row 1 is large; a per-row bound must separate them.
    """
    x = torch.zeros(2, 1024, dtype=torch.float32)
    x[0, :] = 1e-3  # small terms -> small bound
    x[1, :] = 1e3  # large terms -> large bound
    bound = reduction_error_bound(x, torch.float32)
    assert bound[0] < bound[1] / 1e5, (
        f"the bound did not scale with the row's own terms: {bound[0].item():.3e} vs "
        f"{bound[1].item():.3e}"
    )


def test_reduction_bound_sequential_depth_is_the_loosest_and_is_reachable():
    """``parallel_lanes=1`` recovers the fully sequential bound, and is strictly the loosest.

    The escape hatch has to work: a caller whose reduction really is serial, or who simply does not
    know its width, must be able to get the conservative form. Monotonicity is the property that
    makes the parameter safe to reason about -- more parallelism can only tighten the bound, so an
    UNDER-stated ``parallel_lanes`` can never false-fail.
    """
    x = torch.randn(8, 4096, dtype=torch.float32)
    seq = reduction_error_bound(x, torch.float32, parallel_lanes=1)
    warp = reduction_error_bound(x, torch.float32, parallel_lanes=32)
    block = reduction_error_bound(x, torch.float32, parallel_lanes=256)
    assert (seq > warp).all() and (warp > block).all(), (
        "the bound must shrink monotonically as parallel_lanes grows; got "
        f"seq={seq.max().item():.3e} warp={warp.max().item():.3e} block={block.max().item():.3e}"
    )


@pytest.mark.parametrize("lanes", [0, -8, 2.5, None])
def test_reduction_bound_refuses_a_nonsensical_lane_count(lanes):
    """A zero, negative or non-integer width makes the depth nonsensical, not merely loose."""
    with pytest.raises(ValueError, match=r"parallel_lanes must be a positive int"):
        reduction_error_bound(torch.randn(4, 128), torch.float32, parallel_lanes=lanes)
