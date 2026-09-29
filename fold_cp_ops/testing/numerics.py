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
"""Numeric gates for kernel tests, strongest first.

**Why this module exists.** The gate it replaces was one scalar for a whole tensor::

    err = (D.float() - ref).abs().max().item()
    assert err / ref.abs().max().item() < 0.02

Three things that hides, all of them real failure modes:

1. **One catastrophically wrong element is indistinguishable from uniform 2% drift.** The bound is
   set by the *largest* reference magnitude, so an element that cancels to near zero is compared
   against the whole tensor's slack -- it can be wrong by 100% of its own value and pass.
2. **A NaN cannot fail it.** ``nan < 0.02`` is False, so the assert fires -- but only if the NaN is
   the max; a NaN elsewhere makes ``.max()`` NaN and the comparison False, which *does* fail, while
   an Inf/NaN pair can cancel. Neither case is diagnosed, and the message names no coordinate.
3. **An unwritten output region passes** whenever the pre-existing buffer contents happen to be
   close, which for a zeroed buffer and a near-zero reference they are.

The four gates below are ordered by strength. Prefer the strongest one the situation admits.

======================  ==========================================================================
:func:`assert_bitwise`  exact equality; for pure data movement and for integer-exact GEMM
:func:`assert_gemm_exact`  integer operands chosen so the fp32 accumulator is provably exact
:func:`assert_gemm_close`  per-element analytic bound for random operands
:func:`assert_written`  every output element was actually stored
======================  ==========================================================================

**The integer-exactness argument, since the strongest gate rests on it.** Let ``A`` and ``B`` hold
integers with ``|a| <= P`` and ``|b| <= Q``, each exactly representable in the operand dtype. Every
product is an integer with ``|ab| <= PQ``; every partial sum of ``K`` of them is an integer with
``|sum| <= K*P*Q``. fp32 represents every integer up to ``2**24`` exactly, so if

    K * P * Q < 2**24

then *every* partial sum is exact, rounding is the identity at every step, and the accumulator is
bit-identical to an exact integer reference -- **whatever order the terms were summed in**.

That order-independence is the whole point. Tile shape, k-split, cluster, pingpong and accumulator
reuse all change the summation order, so any bound that depends on order cannot gate them
uniformly. ``torch.equal`` can, at every point of the matrix at once.
"""

import math
from typing import NamedTuple

import torch

from fold_cp_ops.testing.numeric_guard import record_assertion

#: Bits of precision (mantissa + implicit leading 1) per operand dtype, and hence the largest
#: integer each represents exactly: ``2**precision``. fp16's 11 bits are why it is the preferred
#: probe dtype -- an 8x wider exact-integer window than bf16 at identical code paths.
_PRECISION_BITS = {
    torch.float16: 11,
    torch.bfloat16: 8,
    torch.float32: 24,
    torch.float64: 53,
    torch.float8_e4m3fn: 4,
    torch.float8_e5m2: 3,
}

#: fp32 unit roundoff, the accumulator's. Half an ulp at 1.0 under round-to-nearest.
_U_FP32 = 2.0**-24

#: Largest integer fp32 represents exactly. The ceiling every partial sum must stay under for
#: :func:`assert_gemm_exact`'s argument to hold.
_FP32_EXACT_INT_MAX = 2**24


def exact_int_bound(dtype: torch.dtype) -> int:
    """Largest integer magnitude ``dtype`` represents exactly.

    Args:
        dtype: A floating torch dtype. Must be one of the six in ``_PRECISION_BITS``; anything else
            (an integer dtype, a quantised type) raises rather than guessing a precision, because a
            wrong answer here silently invalidates the exactness argument that
            :func:`assert_gemm_exact` rests on.

    Returns:
        ``2**precision`` -- e.g. 2048 for fp16, 256 for bf16, 16 for fp8 e4m3. Integers with
        magnitude at or below this are exact; the next integer up may not be.

    Raises:
        KeyError: If ``dtype`` is not a known floating type.
    """
    if dtype not in _PRECISION_BITS:
        raise KeyError(
            f"no precision known for {dtype}; add it to _PRECISION_BITS rather than assuming one -- "
            f"the exactness argument in assert_gemm_exact is only valid for a correct value"
        )
    return 2 ** _PRECISION_BITS[dtype]


def max_exact_operand(K: int, dtype: torch.dtype) -> int:
    """Largest operand magnitude ``P`` for which a ``K``-term fp32 dot product stays exact.

    Solves ``K * P**2 < 2**24`` for ``P``, then caps at what ``dtype`` represents exactly. See the
    module docstring for the derivation.

    Args:
        K: The contraction extent. Must be positive; the returned bound shrinks as ``sqrt(1/K)``,
            so a very large K drives it to 1 and eventually to 0, at which point integer-exact
            testing is impossible and the caller should use :func:`assert_gemm_close` instead.
        dtype: The operand dtype, capping the result at :func:`exact_int_bound`.

    Returns:
        The largest usable ``P``, at least 0. **A return of 0 means no non-trivial integer operand
        set is exact at this K** -- callers must check, since building operands in ``[-0, 0]`` would
        produce an all-zero test that passes vacuously.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    p = int((_FP32_EXACT_INT_MAX // K) ** 0.5)
    while p > 0 and K * p * p >= _FP32_EXACT_INT_MAX:
        p -= 1
    return min(p, exact_int_bound(dtype))


def integer_operands(shape, dtype, device, generator=None, bound=None, K=None):
    """Build an operand of small integers, exactly representable in ``dtype``.

    The values are drawn uniformly from ``[-bound, bound]`` **excluding nothing** -- zeros are
    included deliberately, since a kernel that skips a masked lane and one that adds a zero are
    indistinguishable unless zeros appear.

    Args:
        shape: The tensor shape.
        dtype: Operand dtype. Determines the exact-integer ceiling.
        device: Where to allocate.
        generator: Optional ``torch.Generator`` for reproducibility. Sharing one generator across
            A and B is fine and is what the callers do.
        bound: Operand magnitude ``P``. Defaults to :func:`max_exact_operand` for ``K``. Passing a
            value above :func:`exact_int_bound` raises -- silently rounding an operand would break
            the exactness argument in a way no assertion downstream could detect.
        K: The contraction extent, used only to default ``bound``. Required when ``bound`` is None.

    Returns:
        A tensor of ``dtype`` holding integer values in ``[-bound, bound]``.

    Raises:
        ValueError: If neither ``bound`` nor ``K`` is given, or if ``bound`` exceeds what ``dtype``
            represents exactly, or if the resolved bound is 0 (which would make an all-zero operand
            and a vacuous test).
    """
    if bound is None:
        if K is None:
            raise ValueError("integer_operands needs either an explicit bound or K to derive one")
        bound = max_exact_operand(K, dtype)
    if bound > exact_int_bound(dtype):
        raise ValueError(
            f"bound {bound} exceeds what {dtype} represents exactly ({exact_int_bound(dtype)}); "
            f"the operands would be rounded and the exactness argument would not hold"
        )
    if bound < 1:
        raise ValueError(
            f"resolved operand bound is {bound}: K is too large for an integer-exact test at "
            f"{dtype}. Use assert_gemm_close (the analytic elementwise bound) instead."
        )
    hi = int(bound)
    vals = torch.randint(-hi, hi + 1, tuple(shape), device=device, generator=generator)
    return vals.to(dtype)


def _describe(name, t):
    """One-line shape/dtype/stride summary, for assertion messages.

    Args:
        name: How to refer to the tensor.
        t: The tensor.

    Returns:
        A string such as ``"D (1, 256, 256) torch.float16 stride=(65536, 256, 1)"``.
    """
    return f"{name} {tuple(t.shape)} {t.dtype} stride={tuple(t.stride())}"


def _check_finite(name, t):
    """Raise if ``t`` holds any NaN or Inf, naming how many and where the first one is.

    Separated from the numeric comparison on purpose: a NaN makes ``max()``-based comparisons
    return False rather than True, so it reads as a *pass* in some formulations and as an
    unexplained failure in others. Diagnosing it first makes the message say "NaN", not "0.03 > 0.02".

    Args:
        name: How to refer to the tensor in the message.
        t: The tensor to check. Any dtype; it is viewed as fp32 for the test.

    Returns:
        None.

    Raises:
        AssertionError: If any element is not finite.
    """
    f = t.float()
    bad = ~torch.isfinite(f)
    n = int(bad.sum())
    if n:
        idx = tuple(int(i) for i in torch.nonzero(bad)[0])
        raise AssertionError(
            f"{_describe(name, t)} has {n} non-finite element(s) "
            f"({int(torch.isnan(f).sum())} NaN, {int(torch.isinf(f).sum())} Inf); "
            f"first at {idx} = {f[idx].item()}"
        )


#: Same-width signed integer type per float dtype, for reinterpreting a tensor as its bit pattern.
_BIT_VIEW = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
    torch.float8_e4m3fn: torch.int8,
    torch.float8_e5m2: torch.int8,
}


def _as_bits(t):
    """Reinterpret a tensor's storage as same-width integers, for exact pattern comparison.

    Args:
        t: Any tensor. Made contiguous first, since ``view`` on a strided tensor raises; the copy
            preserves bit patterns exactly, so the comparison is unaffected.

    Returns:
        An integer tensor of the same shape whose values are ``t``'s raw bits. Integer inputs are
        returned as-is.
    """
    if t.dtype not in _BIT_VIEW:
        return t
    return t.contiguous().view(_BIT_VIEW[t.dtype])


def assert_bitwise(actual, expected, what="output"):
    """Assert two tensors have identical **bit patterns**. The strongest gate available.

    Use for anything that is **pure data movement** -- a TMA load staged into SMEM and copied back,
    an identity epilogue, a permuted view -- where the correct result is not "close to" the input
    but *is* the input. Also used by :func:`assert_gemm_exact` for integer-exact GEMM.

    **Bit patterns, not values, and the difference matters in both directions:**

    * ``-0.0 == 0.0`` is True, so a value comparison cannot see a copy path that loses the sign of
      zero -- which flips the sign of a later division. This gate catches it.
    * ``NaN != NaN`` is True, so a value comparison *spuriously fails* on a copy that faithfully
      preserved a NaN. This gate passes it, correctly: the mover's job is to move bits.

    Args:
        actual: The tensor under test.
        expected: What it must equal. Must match ``actual`` in shape and dtype; a mismatch is
            reported as such rather than being broadcast into a confusing element comparison.
        what: A short label naming what is being compared, used in the message.

    Returns:
        None.

    Raises:
        AssertionError: On any shape, dtype, or bit difference. The message gives the count of
            differing elements, the fraction, and the first three coordinates with both values and
            both raw bit patterns -- a located element is far more diagnosable than a statistic,
            and the hex is what distinguishes a sign-of-zero bug from a real numeric one.
    """
    record_assertion("assert_bitwise")
    assert actual.shape == expected.shape, (
        f"{what}: shape mismatch -- {_describe('actual', actual)} vs "
        f"{_describe('expected', expected)}"
    )
    assert actual.dtype == expected.dtype, (
        f"{what}: dtype mismatch -- {actual.dtype} vs {expected.dtype}. Bitwise comparison is only "
        f"meaningful between identical types; cast deliberately if that is what you meant."
    )
    a_bits, e_bits = _as_bits(actual), _as_bits(expected)
    diff = a_bits != e_bits
    n = int(diff.sum())
    if not n:
        return
    total = actual.numel()
    width = a_bits.element_size() * 2  # hex digits
    detail = "; ".join(
        f"{tuple(int(i) for i in c)}: actual={actual[tuple(c)].item()!r} "
        f"(0x{int(a_bits[tuple(c)]) & (2 ** (4 * width) - 1):0{width}x}) "
        f"expected={expected[tuple(c)].item()!r} "
        f"(0x{int(e_bits[tuple(c)]) & (2 ** (4 * width) - 1):0{width}x})"
        for c in torch.nonzero(diff)[:3]
    )
    raise AssertionError(
        f"{what}: {n}/{total} elements ({100.0 * n / total:.3f}%) differ bitwise. "
        f"{_describe('actual', actual)}. First: {detail}"
    )


def gemm_reference(A, B):
    """The exact fp64 reference for ``D = A @ B^T``, batched over leading dims.

    Computed from the **already-rounded** operands, so it carries zero input error: ``A.double()``
    is exact for any 16- or 8-bit input. The error this reference is compared against is therefore
    entirely the accumulator's and the output store's, which is what the bound in
    :func:`gemm_error_bound` models.

    Args:
        A: ``(..., M, K)``. Any float dtype.
        B: ``(..., N, K)``. Same batch dims and same K as ``A``; the contraction is over the LAST
            axis of both, matching the kernel's ``B``-transposed convention.

    Returns:
        An fp64 tensor of shape ``(..., M, N)``.
    """
    return A.double() @ B.double().transpose(-1, -2)


def gemm_error_bound(A, B, out_dtype):
    """Per-element error bound for ``D = A @ B^T`` accumulated in fp32 and stored as ``out_dtype``.

    **This returns a tensor, not a scalar, and that is the point.** The bound for element ``(i, j)``
    is proportional to ``sum_k |A[i,k] * B[j,k]|`` -- the sum of *absolute* products, which is
    exactly the quantity a max-error-over-max-reference ratio throws away. An output element whose
    terms cancel to near zero gets a correspondingly tight bound here, where the scalar gate handed
    it the entire tensor's slack.

    Two contributions:

    * **accumulation** -- ``gamma_K * sum_k |a*b|`` with ``gamma_K = K*u / (1 - K*u)``, ``u = 2**-24``.
      This is the standard sequential-summation bound, so it is *pessimistic* for WGMMA's tree
      order; that direction is the safe one.
    * **the output store** -- one half-ulp of the output dtype, applied to the accumulator's own
      magnitude.

    Args:
        A: ``(..., M, K)`` operand.
        B: ``(..., N, K)`` operand.
        out_dtype: The dtype ``D`` is stored as. An fp32 output contributes a negligible store
            term; a bf16 output contributes a large one, which is why the same kernel needs a
            looser bound at bf16 and it is not a sign of a worse kernel.

    Returns:
        An fp64 tensor of shape ``(..., M, N)``: the maximum absolute deviation each element may
        show. Always strictly positive where any term is non-zero, so a ratio against it is safe.

    Raises:
        ValueError: If ``K * u >= 1``, where the ``gamma`` formula has no meaning. That needs
            ``K > 2**24``, far past anything this kernel runs.
    """
    K = A.shape[-1]
    if K * _U_FP32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    gamma = (K * _U_FP32) / (1.0 - K * _U_FP32)
    abs_terms = A.double().abs() @ B.double().abs().transpose(-1, -2)
    acc_err = gamma * abs_terms
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    # The store rounds the ACCUMULATOR, whose magnitude is at most |ref| + acc_err; bounding |ref|
    # by abs_terms (true, since |sum| <= sum|.|) keeps this a single expression.
    store_err = half_ulp * (abs_terms + acc_err)
    return acc_err + store_err


def reduction_reference(x):
    """The exact fp64 reference for a row sum ``out[m] = sum_n x[m, n]``.

    Computed from the **already-rounded** input, so it carries zero input error: ``x.double()`` is
    exact for any 16- or 32-bit float. The error it is compared against is therefore entirely the
    kernel accumulator's, which is what :func:`reduction_error_bound` models.

    **Use this rather than ``x.float().sum(-1)``.** An fp32 reference is a second approximation with
    its OWN accumulation error, so a bound modelling one reduction would be describing two -- and
    the natural fix, doubling the bound, hides the asymmetry rather than removing it. In fp64 the
    reference's own error is ~2**-29 of the fp32 one and disappears into the bound's slack.

    Args:
        x: ``(..., K)``. Any float dtype; the reduction is over the LAST axis. Must be finite --
            an inf or NaN propagates to every element of the result rather than to one, which is
            the one input shape where a reduction's failure is unlocalizable.

    Returns:
        An fp64 tensor of shape ``x.shape[:-1]``.
    """
    return x.double().sum(-1)


def reduction_error_bound(x, out_dtype, *, parallel_lanes: int = 32):
    """Per-element error bound for a row sum accumulated in fp32 and stored as ``out_dtype``.

    **The reduction analogue of :func:`gemm_error_bound`, and deliberately the same shape.** The
    bound for row ``m`` is proportional to ``sum_n |x[m, n]|`` -- the sum of ABSOLUTE terms, which
    is exactly what a ratio against ``||ref||`` throws away. A row whose terms cancel to near zero
    gets a correspondingly tight bound here, where a pooled L2 over all rows handed it the whole
    tensor's slack.

    Two contributions, matching `gemm_error_bound` term for term:

    * **accumulation** -- ``gamma_d * sum_n |x|`` with ``gamma_d = d*u / (1 - d*u)``, ``u = 2**-24``
      and ``d`` the accumulation DEPTH implied by ``parallel_lanes`` (see below).
    * **the output store** -- one half-ulp of ``out_dtype`` on the accumulator's own magnitude.

    **Why the depth is a parameter, and why the GEMM's sequential ``gamma_K`` is WRONG here.**
    ``gemm_error_bound`` uses the full sequential depth and calls the pessimism safe. For a GEMM it
    is: ``sum|a*b|`` is comparable to ``|result|``, so a loose bound is still a small multiple of
    the answer. A zero-mean row sum breaks that -- it cancels to ``~sqrt(K)`` while ``sum|x|`` grows
    as ``K``, so the sequential bound outgrows the very defect it should catch. MEASURED on CPU,
    4096 rows per cell, against an fp64 oracle, asking what fraction of rows a silently dropped
    1-lane-in-32 partial is DETECTED on:

    ====== ================= ================== ==================== ====================
    K      detected, lanes=1 detected, lanes=32 headroom MIN (l=32)  headroom MEDIAN
    ====== ================= ================== ==================== ====================
    1024   99.2%             100.0%             87x                  819x
    4096   94.3%             99.8-99.9%         607x                 5158x
    16384  57.3%             98.4-99.0%         4737x                36218x
    32768  10.5%             95.9-96.5%         13403x               98117x
    ====== ================= ================== ==================== ====================

    At ``parallel_lanes=1`` the bound is nearly VACUOUS by K=32768 -- it misses the dropped lane on
    9 rows in 10. The default of 32 assumes only that a WARP's partials reduce in parallel, which
    every reduction in this repo guarantees by construction.

    **Read the headroom columns as what they are.** The MIN is an extreme ORDER STATISTIC over 4096
    rows and swings with the draw -- an independent recomputation on other seeds got 145/1000/4933/
    18770 for the same four K, which is the same quantity and not a discrepancy. The detection
    column is the one that reproduces (spreads above are across five seeds) and it is the one that
    matters. Quoting the min alone reads far tighter than the bound is, so the median is here too.

    The honest summary: a dropped 1-in-32 lane is caught on 96-100% of rows, while a defect ~100x
    smaller at K=1024 passes unnoticed on a typical row. Detection is not 100% and cannot be -- a
    dropped lane's own terms can cancel to near zero on an individual row, which is a property of
    the defect rather than of the bound.

    Args:
        x: ``(..., K)`` input to the reduction, in its ORIGINAL dtype. The reduction is over the
            LAST axis. Passing an already-upcast copy is harmless (``.double()`` is idempotent) but
            passing the OUTPUT instead of the input silently produces a bound from the wrong
            magnitudes, which is the one misuse that still returns a plausible-looking tensor.
        out_dtype: The dtype the row sum is stored as. Must be a key of :data:`_PRECISION_BITS`; an
            unlisted dtype raises ``KeyError`` rather than defaulting, because a guessed precision
            is a silently wrong bound.
        parallel_lanes: How many partial sums the reduction accumulates INDEPENDENTLY before
            combining them in a tree. Must be a positive int. ``1`` is the fully sequential worst
            case and is what a caller passes when it does not know; ``32`` (the default) is one
            warp. **Passing a value LARGER than the kernel really uses under-bounds the error and
            can false-fail a correct kernel**, so raise it only for a reduction whose width is known
            -- a 256-thread block reduce may pass 256, and the bound tightens ~4x.

    Returns:
        An fp64 tensor of shape ``x.shape[:-1]``: the maximum absolute deviation each row sum may
        show. Strictly positive wherever the row has any non-zero term, so a ratio against it is
        safe; an all-zero row gives exactly 0, which demands a bit-exact 0 back and is correct.

    Raises:
        ValueError: If ``K * u >= 1``, where the ``gamma`` formula has no meaning (that needs
            ``K > 2**24``, far past anything these kernels run), or if ``parallel_lanes`` is not a
            positive int -- a zero or negative width would make the depth nonsensical rather than
            merely loose.
        KeyError: If ``out_dtype`` is not a known float dtype.
    """
    K = x.shape[-1]
    if K * _U_FP32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    if not (isinstance(parallel_lanes, int) and parallel_lanes >= 1):
        raise ValueError(f"parallel_lanes must be a positive int, got {parallel_lanes!r}")
    # Depth of a blocked reduction: each lane accumulates its own slice SEQUENTIALLY, then the
    # lanes' partials combine in a tree. At lanes=1 this collapses to K, the sequential bound.
    depth = (
        K
        if parallel_lanes == 1
        else math.ceil(K / parallel_lanes) + math.ceil(math.log2(parallel_lanes))
    )
    gamma = (depth * _U_FP32) / (1.0 - depth * _U_FP32)
    abs_terms = x.double().abs().sum(-1)
    acc_err = gamma * abs_terms
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    # The store rounds the ACCUMULATOR, whose magnitude is at most |ref| + acc_err; bounding |ref|
    # by abs_terms (true, since |sum| <= sum|.|) keeps this a single expression -- same argument as
    # gemm_error_bound's store term.
    store_err = half_ulp * (abs_terms + acc_err)
    return acc_err + store_err


def epilogue_error_bound(A, B, reference, out_dtype, alpha=1.0):
    """Per-element bound for ``alpha * (A @ B^T)`` plus fp32-exact epilogue terms.

    The GEMM bound with two adjustments for an epilogue that scales the accumulator and adds
    further terms (``beta * C``, a row bias, a column bias):

    * the accumulation error is scaled by ``|alpha|``, since the epilogue multiplies the
      accumulator before adding anything;
    * the store term is taken against the FINAL reference, not against the raw product -- the
      output that gets rounded is the post-epilogue value.

    The added terms themselves contribute nothing extra: ``C`` and the biases enter as exact fp32
    values and their fp32 additions are dominated by the store rounding that follows.

    Args:
        A: ``(..., M, K)`` operand.
        B: ``(..., N, K)`` operand.
        reference: The exact fp64 reference for the WHOLE epilogue, including every added term.
            Must already have ``alpha`` and ``beta`` applied -- this function scales the *bound*,
            not the reference, and passing an unscaled one silently understates the store term.
        out_dtype: The dtype the result is stored as.
        alpha: The accumulator scale. Its magnitude is what matters; a sign is ignored.

    Returns:
        An fp64 tensor of per-element allowances, shaped like ``reference``.
    """
    K = A.shape[-1]
    gamma = (K * _U_FP32) / (1.0 - K * _U_FP32)
    abs_terms = A.double().abs() @ B.double().abs().transpose(-1, -2)
    acc_err = abs(float(alpha)) * gamma * abs_terms
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    return acc_err + half_ulp * (reference.double().abs() + acc_err)


#: Maximum absolute error of the hardware sigmoid, ``0.5 + 0.5*tanh(0.5*x)`` built on
#: ``tanh.approx.f32``, whose PTX guarantee is ``2**-11`` absolute; the 0.5 scaling halves it. See
#: ``fold_cp_ops._internal.activation``. This dominates every other term in a gated bound at bf16.
_SIGMOID_ABS_ERR = 2.0**-12

#: Maximum of ``|d/dx sigmoid(x)| = sigma*(1-sigma)``, attained at x = 0. Used to carry the gate
#: pre-activation's error THROUGH the sigmoid rather than around it.
_SIGMOID_MAX_SLOPE = 0.25


def gated_error_bound(A, Wg, Wp, reference, out_dtype):
    """Per-element bound for ``sigmoid(A @ Wg^T + bg) * (A @ Wp^T + bp)`` stored as ``out_dtype``.

    **Why the GEMM bound cannot be reused directly.** `gemm_error_bound` bounds a linear result;
    the gate is nonlinear, so the two pre-activations' errors reach the output through different
    paths -- the gate's through the sigmoid's slope and scaled by the up value, the up's directly.
    Applying a linear bound to a gated output is either far too loose (if you bound the 2N
    pre-activation) or unsound (if you bound only one half).

    The derivation, with ``Eg`` and ``Eu`` the two accumulation errors::

        computed = (sigma(g + dg) + eps) * (u + du)          |dg| <= Eg, |du| <= Eu, |eps| <= 2**-12
        exact    = sigma(g) * u

        |computed - exact| <= |sigma(g+dg) - sigma(g)| * |u + du|      (gate error, through sigma')
                            + |eps| * |u + du|                          (hardware sigmoid error)
                            + sigma(g) * |du|                           (up error, sigma <= 1)
                           <= (0.25*Eg + 2**-12) * (|u| + Eu) + Eu

    then one half-ulp of the output dtype on top, taken against the final magnitude.

    Note which term dominates: the ``2**-12 * |u|`` sigmoid term is usually LARGER than either
    accumulation term, so the gate -- not the matmul -- is where a gated kernel's error mostly comes
    from. That is the hardware's approximation showing through rather than a worse kernel, and it is
    why an ULP-scale tolerance fails here while a correct kernel is comfortably inside this bound.

    Args:
        A: ``(..., M, K)`` activation, in any dtype (cast to fp64 internally).
        Wg: ``(..., N, K)`` gate weight. Contracted against `A`'s last axis, matching the kernel.
        Wp: ``(..., N, K)`` up weight. Must have `Wg`'s shape; a mismatch is a broadcast error here
            rather than a wrong bound.
        reference: The exact fp64 reference for the GATED output, ``(..., M, N)`` -- the value the
            store rounds. Must already include the biases and any post-gate mask, or the store term
            is taken against the wrong magnitude.
        out_dtype: The dtype the gated output is stored as.

    Returns:
        An fp64 tensor shaped like `reference`: the maximum absolute deviation each element may
        show. Strictly positive wherever the up projection is non-zero, so a ratio against it is
        safe.

    Raises:
        ValueError: If ``K * u >= 1``, where the gamma formula has no meaning.
    """
    K = A.shape[-1]
    if K * _U_FP32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    gamma = (K * _U_FP32) / (1.0 - K * _U_FP32)
    a64 = A.double().abs()
    # Accumulation errors of the two pre-activations. No store term: neither is written out -- both
    # stay in fp32 registers until the gate consumes them.
    e_gate = gamma * (a64 @ Wg.double().abs().transpose(-1, -2))
    e_up = gamma * (a64 @ Wp.double().abs().transpose(-1, -2))
    # |u| is bounded by the sum of absolute products, which is what the reference cannot supply
    # (the gate has already scaled it by sigma <= 1, so the reference UNDERSTATES |u|).
    abs_up = a64 @ Wp.double().abs().transpose(-1, -2)
    gate_term = (_SIGMOID_MAX_SLOPE * e_gate + _SIGMOID_ABS_ERR) * (abs_up + e_up)
    pre_store = gate_term + e_up
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    return pre_store + half_ulp * (reference.double().abs() + pre_store)


def sigmoid_error_bound(A, W3, reference, out_dtype):
    """Per-element bound for a plain ``sigmoid(A @ W3^T + b3)`` stored as `out_dtype`.

    The un-folded sibling of :func:`gated_error_bound`: one pre-activation, one sigmoid, no pairing.
    Its accumulation error reaches the output only through the sigmoid's slope, so the allowance is
    far tighter than a gated one -- and it has to be, since the output lives in ``(0, 1)`` where a
    loose bound would accept nearly anything.

    Args:
        A: ``(..., M, K)`` activation, in any dtype (cast to fp64 internally).
        W3: ``(..., N3, K)`` weight, contracted against `A`'s last axis.
        reference: The exact fp64 ``(..., M, N3)`` reference, bias included -- the value the store
            rounds.
        out_dtype: The dtype the result is stored as.

    Returns:
        An fp64 tensor shaped like `reference`. Strictly positive everywhere: the hardware sigmoid's
        own ``2**-12`` is a floor, so a ratio against it is always safe.

    Raises:
        ValueError: If ``K * u >= 1``, where the gamma formula has no meaning.
    """
    K = A.shape[-1]
    if K * _U_FP32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    gamma = (K * _U_FP32) / (1.0 - K * _U_FP32)
    e_pre = gamma * (A.double().abs() @ W3.double().abs().transpose(-1, -2))
    pre_store = _SIGMOID_MAX_SLOPE * e_pre + _SIGMOID_ABS_ERR
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    return pre_store + half_ulp * (reference.double().abs() + pre_store)


def fused_ln_gated_error_bound(x, norm_weight, norm_bias, Wg, Wp, reference, eps, out_dtype):
    """Per-element bound for ``sigmoid(LN(x)@Wg^T + bg) * (LN(x)@Wp^T + bp)`` stored as `out_dtype`.

    **A tensor, not a scalar, for the same reason as every other bound here**: an output element
    whose products cancel to near zero gets a correspondingly tight allowance, where a
    max-error-over-max-reference ratio would hand it the whole tensor's slack. That is precisely the
    element a fused kernel is most likely to get wrong.

    **What the fusion adds over `gated_error_bound`.** The kernel does not multiply the exact
    ``LN(x)``. It normalizes in fp32 from fp32 statistics and writes the result back in the
    ACTIVATION's dtype before the MMA, so its A operand is a perturbed ``Ahat``. That perturbation
    is not a rounding of the OUTPUT -- it is a rounding of an INPUT, and it reaches the output
    amplified by the weights. Bounding it as an output tolerance (the tempting shortcut) is unsound
    at large ``|W|`` and far too loose at small.

    So the perturbation is carried through the SAME two paths `gated_error_bound` uses, by widening
    the two pre-activation error terms::

        dA  = |Ahat - LN(x)|  +  rel_stat * |LN(x)|
        Eg  = gamma_K * (|Ahat| @ |Wg|^T)  +  (dA @ |Wg|^T)
        Eu  = gamma_K * (|Ahat| @ |Wp|^T)  +  (dA @ |Wp|^T)

    and then reusing that derivation verbatim: the gate's error enters through the sigmoid's slope
    scaled by the up magnitude, the up's enters directly, the hardware sigmoid contributes its own
    ``2**-12``, and the store contributes one half-ulp of `out_dtype`.

    ``rel_stat`` covers the statistics' own fp32 error -- ``Sum x`` and ``Sum x^2`` each accumulate
    ``K`` terms (``gamma_K``), ``rsqrt`` is the fast-math approximation (``2**-22``), and the
    normalize itself is a few fp32 operations. ``Ahat`` as modelled below ALREADY carries an fp32
    statistics error of that order in an unknown direction, so adding the term again double-counts
    it -- deliberately, since that is the conservative direction and the alternative is a bound that
    can be violated by a correct kernel.

    **The one input regime this does NOT bound**: a row that is nearly constant. The kernel computes
    the variance as ``E[x^2] - mu^2``, a catastrophic cancellation there, so its RELATIVE error is
    unbounded by ``gamma_K`` -- which models the error of a SUM, not of a difference of two nearly
    equal sums. ``rstd`` is then enormous and amplifies it into every output. Measured: an fp16
    input whose per-row offset reached ~330 (where fp16's ulp is 0.25, so the row quantizes to a
    constant) exceeded this bound by 12.8x on a kernel that is bit-identical to its reference
    implementation. That is an ill-conditioned INPUT, not a defect. A caller who must gate such a row
    needs a two-pass (centred) reference and a bound derived from it; a caller who does not should
    keep the per-row means bounded, which is what the test harnesses here do.

    Args:
        x: ``(M, K)`` un-normalized activation, in the kernel's operand dtype. Its dtype decides the
            width ``Ahat`` is modelled at, so passing an fp32 copy silently tightens the bound past
            what the kernel can meet.
        norm_weight: ``(K,)`` fp32 LayerNorm gain.
        norm_bias: ``(K,)`` fp32 LayerNorm bias, or None.
        Wg: ``(N, K)`` gate weight. Contracted against the last axis, matching the kernel.
        Wp: ``(N, K)`` up weight. Must have `Wg`'s shape.
        reference: The exact fp64 reference for the GATED output, ``(M, N)`` -- the value the store
            rounds. Must already include the biases and any post-gate mask, or the store term is
            taken against the wrong magnitude.
        eps: The LayerNorm variance floor the kernel was given. Must be the same value, or ``Ahat``
            is modelled against a different normalization than the one that ran.
        out_dtype: The dtype the gated output is stored as.

    Returns:
        An fp64 tensor shaped like `reference`: the maximum absolute deviation each element may
        show. Strictly positive wherever the up projection is non-zero.

    Raises:
        ValueError: If ``K * u >= 1``, where the gamma formula has no meaning.
    """
    K = x.shape[-1]
    if K * _U_FP32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    gamma = (K * _U_FP32) / (1.0 - K * _U_FP32)
    nb64 = None if norm_bias is None else norm_bias.double()
    an = torch.nn.functional.layer_norm(x.double(), (K,), norm_weight.double(), nb64, eps)
    # The kernel's ACTUAL A operand: normalized in fp32, stored back at the activation's width.
    ahat = torch.nn.functional.layer_norm(
        x.float(),
        (K,),
        norm_weight.float(),
        None if norm_bias is None else norm_bias.float(),
        eps,
    ).to(x.dtype)
    rel_stat = gamma + 2.0**-22 + 4.0 * _U_FP32
    dA = (ahat.double() - an).abs() + rel_stat * an.abs()
    a64 = ahat.double().abs()
    wg64 = Wg.double().abs().transpose(-1, -2)
    wp64 = Wp.double().abs().transpose(-1, -2)
    e_gate = gamma * (a64 @ wg64) + (dA @ wg64)
    e_up = gamma * (a64 @ wp64) + (dA @ wp64)
    # |u| is bounded by the sum of absolute products over the PERTURBED operand's envelope; the
    # reference cannot supply it (the gate has already scaled it by sigma <= 1).
    abs_up = (a64 + dA) @ wp64
    gate_term = (_SIGMOID_MAX_SLOPE * e_gate + _SIGMOID_ABS_ERR) * (abs_up + e_up)
    pre_store = gate_term + e_up
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    return pre_store + half_ulp * (reference.double().abs() + pre_store)


def fused_ln_sigmoid_error_bound(x, norm_weight, norm_bias, W3, reference, eps, out_dtype):
    """Per-element bound for the fused output gate ``sigmoid(LN(x) @ W3^T + b3)``.

    :func:`sigmoid_error_bound` widened by the fusion's own A-perturbation, exactly as
    :func:`fused_ln_gated_error_bound` widens :func:`gated_error_bound`: one pre-activation, one
    sigmoid, no pairing. Its error reaches the output only through the sigmoid's slope, so the bound is much
    tighter than the gated one -- and it should be, since the output is confined to ``(0, 1)`` and a
    loose bound there would accept almost anything.

    Args:
        x: ``(M, K)`` un-normalized activation, in the kernel's operand dtype.
        norm_weight: ``(K,)`` fp32 LayerNorm gain.
        norm_bias: ``(K,)`` fp32 LayerNorm bias, or None.
        W3: ``(N3, K)`` output-gate weight.
        reference: The exact fp64 ``(M, N3)`` reference, biases included.
        eps: The LayerNorm variance floor the kernel was given.
        out_dtype: The dtype the gate output is stored as.

    Returns:
        An fp64 tensor shaped like `reference`.

    Raises:
        ValueError: If ``K * u >= 1``.
    """
    K = x.shape[-1]
    if K * _U_FP32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    gamma = (K * _U_FP32) / (1.0 - K * _U_FP32)
    nb64 = None if norm_bias is None else norm_bias.double()
    an = torch.nn.functional.layer_norm(x.double(), (K,), norm_weight.double(), nb64, eps)
    ahat = torch.nn.functional.layer_norm(
        x.float(),
        (K,),
        norm_weight.float(),
        None if norm_bias is None else norm_bias.float(),
        eps,
    ).to(x.dtype)
    rel_stat = gamma + 2.0**-22 + 4.0 * _U_FP32
    dA = (ahat.double() - an).abs() + rel_stat * an.abs()
    w64 = W3.double().abs().transpose(-1, -2)
    e_pre = gamma * (ahat.double().abs() @ w64) + (dA @ w64)
    pre_store = _SIGMOID_MAX_SLOPE * e_pre + _SIGMOID_ABS_ERR
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    return pre_store + half_ulp * (reference.double().abs() + pre_store)


class _AlgFoldTerms(NamedTuple):
    """The per-ROW quantities every algebraic-fold bound needs, computed once.

    Attributes:
        gamma: The fp32 summation growth factor for a length-`K` reduction.
        x64, absx: The activation and its magnitude, fp64.
        w64, b64: The LayerNorm gain and bias, fp64; `b64` is None when there is no bias.
        r64, s64: The EXACT ``rstd`` and ``rstd*mu``, fp64.
        dr, ds: Bounds on how far the KERNEL's own ``r`` and ``s`` sit from those, including its
            fast-math ``rsqrt`` and its reduction order.
        an: The exact ``LN(x)``, fp64, for forming a reference pre-activation.
    """

    gamma: float
    x64: object
    absx: object
    w64: object
    b64: object
    r64: object
    s64: object
    dr: object
    ds: object
    an: object


def _alg_fold_terms(x, norm_weight, norm_bias, eps):
    """Compute the row statistics an algebraic-fold bound needs, both exactly and as the kernel does.

    Purpose
        Shared by the dual bound and the output-gate bound. Both repair the SAME rows with the SAME
        ``(r, s)``, so computing these twice would be two chances for the two bounds to disagree
        about the kernel they describe.

    Semantics
        Forms each statistic twice -- once in the kernel's arithmetic (fp32, ``var = E[x^2] - mu^2``)
        and once exactly (fp64) -- and takes the measured gap as the statistic's error, widened by
        `rel_stat` for what the comparison cannot see: the kernel's reduction ORDER differs from
        torch's, and its ``rsqrt`` is the fast-math approximation.

    Args:
        x: ``(M, K)`` un-normalized activation in the kernel's operand dtype. ``K`` must satisfy
            ``K * u < 1`` or the gamma factor is meaningless.
        norm_weight: ``(K,)`` fp32 gain.
        norm_bias: ``(K,)`` fp32 bias, or None.
        eps: The variance floor the kernel was given. Must be the SAME value; a different one moves
            ``r`` and the bound describes a different kernel.

    Returns:
        An `_AlgFoldTerms`.

    Raises:
        ValueError: If ``K * u >= 1``, or if any row's fp32 ``E[x^2] - mu^2 + eps`` is non-positive
            -- the nearly-constant-row regime this bound does not model, where a NaN allowance
            would be reported as a kernel defect rather than an ill-conditioned input.
    """
    K = x.shape[-1]
    if K * _U_FP32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    gamma = (K * _U_FP32) / (1.0 - K * _U_FP32)
    x64 = x.double()
    w64 = norm_weight.double()
    b64 = None if norm_bias is None else norm_bias.double()
    xf = x.float()
    mu_k = xf.sum(-1, keepdim=True) / K
    var_k = (xf * xf).sum(-1, keepdim=True) / K - mu_k * mu_k
    if not bool((var_k + eps > 0).all()):
        bad = int((var_k + eps <= 0).nonzero()[0, 0])
        raise ValueError(
            f"row {bad}: fp32 E[x^2]-mu^2 + eps is non-positive, so the kernel's own rstd is "
            "undefined. This is the nearly-constant-row regime the bound does not model -- returning "
            "a NaN allowance here would report it as a kernel defect instead of an ill-conditioned "
            "input. Use a centred (two-pass) reference for such a row."
        )
    r_k = torch.rsqrt(var_k + eps).double()
    s_k = (r_k * mu_k.double()).double()
    mu64 = x64.mean(-1, keepdim=True)
    r64 = torch.rsqrt(x64.var(-1, keepdim=True, unbiased=False) + eps)
    s64 = r64 * mu64
    rel_stat = gamma + 2.0**-22 + 4.0 * _U_FP32
    return _AlgFoldTerms(
        gamma=gamma,
        x64=x64,
        absx=x64.abs(),
        w64=w64,
        b64=b64,
        r64=r64,
        s64=s64,
        dr=(r_k - r64).abs() + rel_stat * r64,
        ds=(s_k - s64).abs() + rel_stat * s64.abs(),
        an=torch.nn.functional.layer_norm(x64, (K,), w64, b64, eps),
    )


def _alg_fold_preact_error(t, norm_weight, W, bias_vec, colsum_rounded=False):
    """Error and magnitude of ONE projection's pre-activation under the algebraic fold.

    Purpose
        The fold's arithmetic is per-PROJECTION, so this is the whole of it: the dual bound calls it
        twice (gate, up) and the output-gate bound once. Sharing it is what keeps the gate's bound
        describing the same fusion the dual's does.

    Semantics
        The accumulator term is a signed TRANSPORT, not a triangle-inequality bound: the exact
        effect the fold's rounding has on THIS `x` is computed, because bounding it would be loose
        by orders of magnitude. ``c`` reduces the fold BEFORE rounding (the kernel's inherited
        convention), so its only error is the summation's. The correction's four fp32 operations act
        on the ABSOLUTE terms, which is where this fusion's conditioning lives -- they can be far
        larger than their difference.

    Args:
        t: The `_AlgFoldTerms` for this `x`.
        norm_weight: ``(K,)`` fp32 gain, needed at fp32 to reproduce the STORED fold exactly.
        W: ``(N, K)`` weight for this projection, in the kernel's operand dtype.
        bias_vec: ``(N,)`` projection bias, or None.
        colsum_rounded: Which ``c`` convention the kernel used -- False reduces the fold BEFORE it
            is rounded into the weight dtype (the dual's), True after (the output gate's). Must
            match `build_folded_dual_operands` / `build_folded_gate3_operands` for the call being
            bounded: they differ in WHICH rounding error cancels, so the wrong one is not merely
            loose, it attributes the error to the wrong term.

    Returns:
        ``(err, mag)``: the ``(M, N)`` fp64 bound on this projection's pre-activation error, and the
        ``(M, N)`` fp64 magnitude of the exact pre-activation including its bias.
    """
    W64 = W.double()
    wW = t.w64.unsqueeze(0) * W64  # the exact fold, (N, K)
    bwhat = (norm_weight.float().unsqueeze(0) * W.float()).to(W.dtype).double()  # as stored
    gt = t.x64 @ wW.transpose(-1, -2)
    e_acc = (
        t.gamma * (t.absx @ bwhat.abs().transpose(-1, -2))
        + (t.x64 @ bwhat.transpose(-1, -2) - gt).abs()
    )
    if colsum_rounded:
        # `c` reduces what the MMA reads, so the fold's rounding is transported EXACTLY here too --
        # a signed quantity, not a bound. It is precisely this term that the unrounded convention
        # leaves in `e_acc` uncancelled, which is why the rounded one is flat in |mu|/sigma.
        ct = bwhat.sum(-1)
        e_c = t.gamma * bwhat.abs().sum(-1) + (bwhat.sum(-1) - wW.sum(-1)).abs()
    else:
        ct = wW.sum(-1)
        e_c = t.gamma * wW.abs().sum(-1)
    if t.b64 is None:
        dvec = torch.zeros_like(ct)
        e_d = torch.zeros_like(ct)
    else:
        dvec = (t.b64.unsqueeze(0) * W64).sum(-1)
        e_d = t.gamma * (t.b64.abs().unsqueeze(0) * W64.abs()).sum(-1)
    terms = t.r64 * gt.abs() + (t.s64 * ct).abs() + dvec.abs()
    err = (
        t.r64 * e_acc
        + t.dr * gt.abs()
        + t.s64.abs() * e_c
        + t.ds * ct.abs()
        + e_d
        + 4.0 * _U_FP32 * terms
    )
    exact = t.an @ W64.transpose(-1, -2)
    if bias_vec is not None:
        exact = exact + bias_vec.double()
    return err, exact.abs()


def alg_fold_sigmoid_error_bound(x, norm_weight, norm_bias, W3, reference, eps, out_dtype, b3=None):
    """Per-element bound for ``sigmoid(LN(x) @ W3^T + b3)`` under the ALGEBRAIC fold.

    The output gate's counterpart to :func:`fused_ln_sigmoid_error_bound`, and NOT interchangeable
    with it: that one models a rounding of the normalized ACTIVATION, which this fusion never
    performs, while this models a rounding of the WEIGHT, which that one never performs. Using
    either for the other is unsound rather than merely loose.

    The pre-activation error is the same per-projection quantity the dual bound uses, reached
    through the same helper; only its passage to the output differs. Here it goes through one
    sigmoid, so it is multiplied by the maximum slope (1/4) rather than by the other half's
    magnitude -- which is why this bound is far tighter, and why it must be: the output lives in
    ``(0, 1)``, where a loose bound would accept nearly anything.

    Args:
        x: ``(M, K)`` un-normalized activation, in the kernel's operand dtype.
        norm_weight: ``(K,)`` fp32 LayerNorm gain.
        norm_bias: ``(K,)`` fp32 LayerNorm bias, or None. Must match what the kernel was given: its
            ABSENCE removes a term rather than zeroing one.
        W3: ``(N3, K)`` output-gate weight, same dtype as `x`.
        reference: The exact fp64 ``(M, N3)`` reference, bias included.
        eps: The LayerNorm variance floor the kernel was given.
        out_dtype: The dtype the gate output is stored as.
        b3: ``(N3,)`` output-gate bias, or None.

    Returns:
        An fp64 tensor shaped like `reference`.

    Raises:
        ValueError: From `_alg_fold_terms` -- if ``K * u >= 1`` or a row is nearly constant.
    """
    t = _alg_fold_terms(x, norm_weight, norm_bias, eps)
    # ``colsum_rounded=True``: the gate's fold reduces the ROUNDED weight, the opposite of the
    # dual's convention, matching `build_folded_gate3_operands`' default and the upstream it
    # reproduces.
    e_pre, _ = _alg_fold_preact_error(t, norm_weight, W3, b3, colsum_rounded=True)
    pre_store = _SIGMOID_MAX_SLOPE * e_pre + _SIGMOID_ABS_ERR
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    return pre_store + half_ulp * (reference.double().abs() + pre_store)


def alg_fold_xgate_error_bound(
    x, x_gate, norm_weight, norm_bias, Wg, Wp, reference, eps, out_dtype, bg=None, bp=None
):
    """Per-element bound for ``sigmoid(x_gate @ Wg^T + bg) * (LN(x) @ Wp^T + bp)``, the TWO-A path.

    Purpose
        The two-A kernel's arms are produced by DIFFERENT arithmetic, so neither existing bound
        describes it. `alg_fold_error_bound` models both arms as folded-and-repaired, which is
        **unsound here in the direction that matters**: the gate arm performs none of that, so that
        bound hands it an allowance for errors it cannot incur, and a gate arm that was genuinely
        wrong could sit inside it. `gated_error_bound` has the opposite problem on the value arm.

    Semantics
        Composed from the two models the kernel actually uses, one per arm:

        * **The gate arm is a PLAIN 16-bit GEMM.** ``x_gate`` arrives normalized, so there is no
          gain folded into ``Wg``, no rank-one repair and no statistic in the path. Its error is the
          accumulation error of `gated_error_bound`'s gate half and nothing else.
        * **The value arm is the algebraic fold**, reached through the same
          `_alg_fold_preact_error` the dual and output-gate bounds use -- with
          ``colsum_rounded=True``, because this path folds its value weight with
          `build_folded_gate3_operands`, whose convention reduces the ROUNDED fold. Passing False
          would model a rounding the kernel does not perform and leave the bound carrying pure
          slack.

        The two then meet exactly as in every gated bound here: the gate's error through the
        sigmoid's maximum slope plus the hardware sigmoid's absolute error, scaled by the value
        magnitude; the value's error directly; one half-ulp of the store on top.

        ``x`` and ``x_gate`` are UNRELATED tensors and both are required. Passing ``x`` for both --
        the shape-compatible mistake -- silently bounds a kernel this one is not.

    Args:
        x: ``(M, K)`` un-normalized VALUE activation, in the kernel's operand dtype. Only the value
           arm normalizes.
        x_gate: ``(M, K)`` PRE-NORMALIZED gate activation, same shape and dtype as `x`. Must be the
            tensor the kernel was given, not ``LN(x)`` recomputed: it is read raw, so its own
            rounding is already in it.
        norm_weight: ``(K,)`` fp32 LayerNorm gain. Applies to the value arm only.
        norm_bias: ``(K,)`` fp32 LayerNorm bias, or None. Its ABSENCE removes a term rather than
            zeroing one, so it must match what the kernel was given.
        Wg: ``(N, K)`` gate weight, RAW -- no gain folded in. Same dtype as `x`.
        Wp: ``(N, K)`` value weight, pre-fold. Must have `Wg`'s shape and dtype.
        reference: The exact fp64 ``(M, N)`` reference for the COMBINED output -- biases and any
            post-gate mask already applied, or the store term is taken against the wrong magnitude.
        eps: The LayerNorm variance floor the kernel was given.
        out_dtype: The dtype the combined output is stored as.
        bg: ``(N,)`` gate projection bias, or None. Sizes the gate magnitude only.
        bp: ``(N,)`` value projection bias, or None. This one matters more -- the value magnitude
            multiplies the gate's error.

    Returns:
        An fp64 tensor shaped like `reference`: the maximum absolute deviation each element may
        show.

    Raises:
        ValueError: If ``K * u >= 1``, where the gamma formula has no meaning; if `Wg` and `Wp`
            disagree in shape or dtype; if `x` and `x_gate` do; or from `_alg_fold_terms` on a
            nearly-constant row, which this bound does not model.
    """
    K = x.shape[-1]
    if K * _U_FP32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    if Wg.shape != Wp.shape or Wg.dtype != Wp.dtype:
        raise ValueError(
            f"Wg{tuple(Wg.shape)}/{Wg.dtype} and Wp{tuple(Wp.shape)}/{Wp.dtype} must match: both "
            "are (N, K) operands of the same kernel at one width"
        )
    if x.shape != x_gate.shape or x.dtype != x_gate.dtype:
        raise ValueError(
            f"x{tuple(x.shape)}/{x.dtype} and x_gate{tuple(x_gate.shape)}/{x_gate.dtype} must "
            "match: the two-A kernel contracts both against the same K at one width"
        )
    # Gate arm: a plain 16-bit GEMM of an already-normalized activation. No fold, no repair.
    gamma = (K * _U_FP32) / (1.0 - K * _U_FP32)
    e_gate = gamma * (x_gate.double().abs() @ Wg.double().abs().transpose(-1, -2))
    # Value arm: the algebraic fold, at the ROUNDED-colsum convention this path folds with.
    t = _alg_fold_terms(x, norm_weight, norm_bias, eps)
    e_up, abs_up = _alg_fold_preact_error(t, norm_weight, Wp, bp, colsum_rounded=True)
    gate_term = (_SIGMOID_MAX_SLOPE * e_gate + _SIGMOID_ABS_ERR) * (abs_up + e_up)
    pre_store = gate_term + e_up
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    return pre_store + half_ulp * (reference.double().abs() + pre_store)


def alg_fold_error_bound(
    x, norm_weight, norm_bias, Wg, Wp, reference, eps, out_dtype, bg=None, bp=None
):
    """Per-element bound for the ALGEBRAIC-FOLD fused LayerNorm dual-gated GEMM.

    Same output as :func:`fused_ln_gated_error_bound` -- ``sigmoid(LN(x)@Wg^T + bg) * (LN(x)@Wp^T +
    bp)`` -- reached by different arithmetic, and therefore needing a different bound. The two are
    NOT interchangeable in either direction, and using one for the other is unsound rather than
    merely loose (see the last section).

    **The arithmetic being bounded.** The ``alg_fold`` fusion never forms ``LN(x)``. It multiplies
    the RAW activation by a folded weight and repairs the difference rank-one in the epilogue::

        LN(x) @ B  =  r*(x @ Bw) - s*c + d
            Bw[p,k] = w_ln[k] * B[k,p]     c[p] = colsum_k Bw[p,k]     d[p] = sum_k b_ln[k]*B[k,p]
            r = rstd,  s = rstd*mu,  both accumulated per row IN the mainloop

    so the error sources are a different set from the prologue-normalize variant's:

    1. **The fold is rounded to the WEIGHT dtype.** ``Bw`` is materialized by the precompute kernel
       as a real ``(2N, K)`` bf16/fp16 tensor, so ``w_ln[k]*B[k,p]`` is rounded once, per element,
       before any MMA sees it. Modelled exactly rather than bounded: the rounded fold is formed here
       and the difference it makes to the product is computed, not estimated.
    2. **The MMA accumulates raw ``x``**, in fp32, over ``K`` terms -- ``gamma_K`` against the
       magnitudes of ``x`` and the ROUNDED fold.
    3. **``c`` and ``d`` are fp32 reductions** over ``K``, and ``c`` sums the fold BEFORE it is
       rounded -- so it does NOT reduce the same numbers the MMA consumes. That asymmetry is the
       kernel's inherited convention, not a modelling choice here (see
       ``_fold_precompute_kernel``), and it costs accuracy: the residual it leaves is proportional to
       raw ``x`` rather than to centred ``x``, which on an offset row is larger by about
       ``|mu|/sigma``. Reducing the rounded fold instead would move a ``|sum dBw|`` term into ``c``
       and remove a strictly larger one from the accumulator.
    4. **``r`` and ``s`` carry the statistics' error**, which then MULTIPLIES the accumulator rather
       than being folded into the operand.
    5. The correction itself is four fp32 operations on quantities that may be much larger than
       their result -- see below.

    **Why this is structurally weaker than the prologue-normalize variant, and where.** The term
    ``r*(x@Bw)`` is proportional to ``|x|``, while the result is proportional to ``|LN(x)|``. When a
    row's mean is large relative to its spread, ``r*(x@Bw)`` and ``s*c`` are two large nearly-equal
    quantities whose difference is small: the fusion computes the answer as a cancellation. Every
    error in ``r``, in the accumulator, and in ``c`` is scaled by the LARGE quantity and lands on the
    SMALL one. That is why the bound below is written in terms of ``r*|Gt| + |s*Ct| + |D|`` -- the
    sum of the correction's absolute terms -- rather than in terms of the result: the ratio between
    those two is the conditioning of this fusion, and it is a property of the input, not a slack.

    ``fused_ln_gated_error_bound`` has no analogue of this term at all, because it normalizes first
    and multiplies second. Applying that bound to ``alg_fold`` would therefore UNDERSTATE the error
    on exactly the inputs where ``alg_fold`` is weakest, which is why the two variants are compared
    against their own bounds and against each other only through those bounds -- never bitwise. They
    are not expected to agree bitwise; the arithmetic order differs by construction.

    **But the comparison goes BOTH ways, and the direction depends on the input.** Measured here at
    ``M=64, N=32, K=128``, as the ratio of this bound's median to `fused_ln_gated_error_bound`'s
    median on the identical input, against the per-row offset added to a unit-variance ``x``:

    ======  ======  ========
    offset  fp16    bf16
    ======  ======  ========
    0       0.23    0.08
    1       0.24    0.09
    10      0.56    0.34
    100     6.67    3.20
    ======  ======  ========

    So ``alg_fold`` is **more accurate than the prologue variant on centred rows** -- by 4x at fp16
    and 13x at bf16 -- and only loses past a crossover near an offset of ~20 sigma. That is not a
    surprise once stated: the prologue variant's dominant term is rounding the NORMALIZED activation
    to 16 bits before the MMA, which ``alg_fold`` never does, while ``alg_fold``'s dominant term is
    the cancellation above, which the prologue variant never has. Neither dominates everywhere.

    Do not read this as a recommendation. The crossover is a property of the input distribution, and
    the workflow's activations are not necessarily centred; the point is only that "the algebraic
    fold is the less accurate one" is false as stated, and a variant choice made on that belief would
    be made on nothing.

    **Tightness.** Against the arithmetic modelled (see ``_simulate_alg_fold`` in the tests), the
    worst element reaches 0.87 (fp16) / 0.95 (bf16) of this bound on centred rows and stays at 0.59 /
    0.88 out at an offset of 100 -- so it is close to attained across the whole range rather than
    decorative, and a pass is real evidence at any offset. It was NOT so before the ``c`` convention
    above was modelled correctly: assuming ``c`` reduced the rounded fold left the bound carrying a
    term the kernel does not incur, and the worst ratio fell to 0.07 at offset 100 -- a bound three
    quarters of which was slack, still passing, and reported as evidence.

    **The regime this does NOT bound**, carried over verbatim from `fused_ln_gated_error_bound`
    because it applies here identically: a nearly-constant row. The kernel forms the variance as
    ``E[x^2] - mu^2``, a catastrophic cancellation whose relative error is not bounded by
    ``gamma_K`` -- which models the error of a SUM, not of a difference of two nearly equal sums.
    ``rstd`` is then enormous and amplifies it into every output. Measured on the prologue variant:
    an fp16 input whose per-row offset reached ~330 (where fp16's ulp is 0.25, so the row quantizes
    to a constant) exceeded that bound by 12.8x on a kernel bit-identical to its reference. That is
    an ill-conditioned INPUT, not a defect. Here the same input is worse still, for the reason in the
    table above -- it is past the crossover in both senses at once. A caller who must gate such a row
    needs a two-pass (centred) reference and a bound derived from it. A row whose fp32 variance goes
    non-positive is refused outright rather than returned as a NaN allowance, since a NaN bound
    reports an ill-conditioned input as a kernel defect.

    **What is modelled exactly vs bounded.** The fold rounding, the ``c`` reduction's operand
    rounding, and the fp32 statistics are computed here in the same form the kernel uses and their
    deviation from the fp64 ideal is measured. Only the fp32 SUMMATION errors (``gamma_K``), the
    ``rsqrt`` fast-math approximation, and the epilogue's own four operations are bounded
    analytically. This costs several fp64 matmuls of the operand's shape, so the function is for
    correctness tests at test shapes, not for production-sized tensors.

    Args:
        x: ``(M, K)`` un-normalized activation, in the kernel's operand dtype. Passing an fp32 copy
            of a bf16 tensor understates the statistics' cancellation and silently tightens the
            bound past what the kernel can meet.
        norm_weight: ``(K,)`` fp32 LayerNorm gain. Its dtype decides nothing here -- it is the
            weight's dtype that the fold rounds to -- but it must be the value the precompute
            kernel was given, or the fold modelled here is not the fold that ran.
        norm_bias: ``(K,)`` fp32 LayerNorm bias, or None. None means the ``d`` term is absent from
            both the kernel and the bound; passing a zero tensor instead is NOT equivalent, since it
            adds a reduction error the kernel never incurs.
        Wg: ``(N, K)`` gate weight, in the dtype the folded ``Bw`` is stored as. That dtype is the
            one source of error term 1, so it must be the kernel's operand dtype and not a widened
            copy.
        Wp: ``(N, K)`` up weight. Must have `Wg`'s shape and dtype.
        reference: The exact fp64 reference for the GATED output, ``(M, N)`` -- the value the store
            rounds. Must already include the biases and any post-gate mask, or the store term is
            taken against the wrong magnitude.
        eps: The LayerNorm variance floor the kernel was given. Must be the same value, or the
            statistics modelled here are not the ones that ran.
        out_dtype: The dtype the gated output is stored as.
        bg: ``(N,)`` gate projection bias, or None. Used only to size the gate's magnitude; omitting
            a bias that WAS applied leaves the bound slightly optimistic in the up magnitude, so
            pass it whenever the kernel had it.
        bp: ``(N,)`` up projection bias, or None. Same as `bg`, and this one matters more -- the up
            magnitude multiplies the gate's error.

    Returns:
        An fp64 tensor shaped like `reference`: the maximum absolute deviation each element may
        show. Strictly positive wherever the up projection is non-zero.

    Raises:
        ValueError: If ``K * u >= 1``, where the gamma formula has no meaning; or if `Wg` and `Wp`
            disagree in shape or dtype, which would mean the two halves of the interleaved fold were
            rounded at different widths -- a condition the kernel cannot represent.
    """
    K = x.shape[-1]
    if K * _U_FP32 >= 1.0:
        raise ValueError(f"K={K} is too large for the gamma bound (K*u >= 1)")
    if Wg.shape != Wp.shape or Wg.dtype != Wp.dtype:
        raise ValueError(
            f"Wg{tuple(Wg.shape)}/{Wg.dtype} and Wp{tuple(Wp.shape)}/{Wp.dtype} must match: the "
            "interleaved fold stores both halves in one tensor, at one width"
        )
    t = _alg_fold_terms(x, norm_weight, norm_bias, eps)
    e_gate, _ = _alg_fold_preact_error(t, norm_weight, Wg, bg)
    e_up, abs_up = _alg_fold_preact_error(t, norm_weight, Wp, bp)
    gate_term = (_SIGMOID_MAX_SLOPE * e_gate + _SIGMOID_ABS_ERR) * (abs_up + e_up)
    pre_store = gate_term + e_up
    half_ulp = 2.0 ** -_PRECISION_BITS[out_dtype]
    return pre_store + half_ulp * (reference.double().abs() + pre_store)


def tolerance_bound(reference, atol=0.0, rtol=0.0):
    """The per-element bound an ``atol``/``rtol`` pair denotes: ``atol + rtol * |reference|``.

    **The conversion path off ``torch.testing.assert_close``, and nothing more.** That function is
    already element-wise -- it is banned (see :mod:`fold_cp_ops.testing.numeric_guard`) because its
    bound is hand-written rather than derived, not because it pools. This builder reproduces its
    tolerance exactly, so a call site moves over with its bar unchanged and any argument about
    whether the bar is right stays a separate, visible question.

    Prefer a DERIVED bound (:func:`gemm_error_bound`, :func:`fused_ln_gated_error_bound`, and the
    rest of this module) wherever one exists: those scale with the operands and the accumulation
    depth, so they stay correct as shapes grow, where a fixed pair silently loosens.

    Args:
        reference: The reference tensor. Only its magnitudes are read; it is not modified. Cast to
            fp64 so the bound does not inherit the reference's own rounding.
        atol: Absolute floor, applied to every element. Must be >= 0 -- a negative floor would make
            the bound smaller than the relative term alone and is a caller bug, so it raises rather
            than producing a quietly over-strict gate.
        rtol: Relative coefficient on ``|reference|``. Must be >= 0, for the same reason.

    Returns:
        An fp64 tensor shaped like ``reference``, ready to pass as :func:`assert_elementwise`'s
        ``bound``. Elements whose reference is zero get exactly ``atol``, so an ``atol=0`` call
        demands a bit-exact match there -- which is the honest reading of ``rtol``-only tolerance,
        and the reason ``assert_close`` defaults ``atol`` above zero.

    Raises:
        ValueError: If ``atol`` or ``rtol`` is negative.
    """
    if atol < 0 or rtol < 0:
        raise ValueError(f"tolerance_bound needs atol >= 0 and rtol >= 0, got {atol=}, {rtol=}")
    return atol + rtol * reference.double().abs()


#: Elements per fp64 slice when :func:`assert_elementwise` splits a large comparison. The working
#: set is roughly six tensors of this size (actual, reference, their difference, the bound, the
#: ratio, the mask), so 2**26 is about 3 GB -- comfortable beside operands that are themselves tens
#: of GB, and large enough that the split never costs anything measurable.
#:
#: **This exists because the comparison OOM'd where the kernel did not.** The A2A-fused token ladder
#: reaches ``(1, 2048, 2048, 512)``: 8.6 GB in fp32, and this function promotes BOTH operands to
#: fp64 for exactness, which is 34 GB before a single temporary. The cell ran, the reference built,
#: and then the *check* died -- outside the test's OOM-skip, so it read as a correctness failure at
#: a shape the kernel had just computed correctly.
_CHUNK_ELEMS = 2**26


def assert_elementwise(actual, reference, bound, what="output", report_top=3):
    """Assert ``|actual - reference| <= bound`` **element by element**, and report properly.

    Large comparisons are split along their LONGEST axis (see :data:`_CHUNK_ELEMS`); the reported
    coordinates are GLOBAL, so the split is invisible except in peak memory.

    Args:
        actual: The tensor under test. Cast to fp64 for the comparison.
        reference: The exact reference, same shape. Typically fp64 from :func:`gemm_reference`.
        bound: Per-element allowance, same shape as ``reference`` (or broadcastable to it). Passing
            a scalar here is legal but throws away this function's reason to exist. A bound that
            varies along the split axis is narrowed with the data; a 0-d, scalar or
            broadcast-along-that-axis bound is passed through, so every legal spelling behaves
            identically split or whole.
        what: Short label for the message.
        report_top: How many worst offenders to list on failure. When the comparison is split, this
            many are reported from the FIRST slice that has any -- enough to localize the defect,
            which is what the coordinates are for.

    Returns:
        The worst observed ``|err| / bound`` ratio, as a float. Callers may print it: a value
        creeping toward 1.0 means the bound is about to become flaky, which is worth knowing
        *before* it fails.

    Raises:
        AssertionError: If ``actual`` holds non-finite values, or if any element exceeds its bound.
            The message gives the violating count and fraction, then the ``report_top`` worst
            offenders with coordinate, actual, reference, bound and ratio.
    """
    record_assertion("assert_elementwise")
    if actual.dim() and actual.numel() > _CHUNK_ELEMS:
        # The LONGEST axis, not the leading one. Splitting axis 0 is the obvious choice and is
        # useless here: the token ladder's output is `(1, 2048, 2048, 512)`, whose leading extent is
        # 1, so a leading-axis split produced exactly one slice of the whole tensor and OOM'd
        # identically. Every tensor this matters for is O(N^2 D) with a batch of 1.
        dim = max(range(actual.dim()), key=lambda d: actual.shape[d])
        per = max(1, actual.numel() // max(1, actual.shape[dim]))
        step = max(1, _CHUNK_ELEMS // per)
        worst = 0.0
        for lo in range(0, actual.shape[dim], step):
            n = min(step, actual.shape[dim] - lo)
            sub = bound
            # `narrow` is a view for a broadcast tensor too, but narrowing an axis the bound does
            # NOT vary along would silently take a prefix of a different axis's values.
            if torch.is_tensor(bound) and bound.dim() == actual.dim():
                if bound.shape[dim] == actual.shape[dim]:
                    sub = torch.narrow(bound, dim, lo, n)
            worst = max(
                worst,
                _assert_elementwise_slice(
                    torch.narrow(actual, dim, lo, n),
                    torch.narrow(reference, dim, lo, n),
                    sub,
                    what,
                    report_top,
                    offset=lo,
                    offset_dim=dim,
                ),
            )
        return worst
    return _assert_elementwise_slice(actual, reference, bound, what, report_top, 0, 0)


def _assert_elementwise_slice(actual, reference, bound, what, report_top, offset, offset_dim):
    """One slice of :func:`assert_elementwise`'s comparison, reporting GLOBAL coordinates.

    Args:
        actual: The slice under test.
        reference: The matching reference slice.
        bound: The allowance, already narrowed if it varies along the split axis.
        what: Short label for the message.
        report_top: How many worst offenders to list.
        offset: What to add to the coordinate on ``offset_dim`` so the reported index refers to the
            WHOLE tensor. A slice-local index would send the reader to the wrong row, which is worse
            than no index at all.
        offset_dim: Which axis was split. 0 when the comparison was not split, where ``offset`` is
            also 0 and the arithmetic is a no-op.

    Returns:
        The worst ``|err| / bound`` ratio in this slice, as a float.

    Raises:
        AssertionError: As :func:`assert_elementwise`.
    """
    _check_finite(what, actual)
    a = actual.double()
    ref = reference.double()
    assert a.shape == ref.shape, (
        f"{what}: shape mismatch -- {tuple(a.shape)} vs reference {tuple(ref.shape)}"
    )
    err = (a - ref).abs()
    # A zero bound (an all-zero row) admits only an exact match; the tiny floor keeps the ratio
    # finite so the report can rank offenders instead of printing inf.
    #
    # The scalar arm is not decoration: `torch.clamp(1e-2, min=...)` is a TypeError, so a plain
    # float bound -- which the signature documents as legal, and which a uniform absolute tolerance
    # genuinely is -- crashed here rather than comparing anything.
    safe = (
        torch.clamp(bound.double(), min=1e-300)
        if torch.is_tensor(bound)
        else max(float(bound), 1e-300)
    )
    ratio = err / safe
    worst = float(ratio.max())
    over = ratio > 1.0
    n = int(over.sum())
    if not n:
        return worst
    total = a.numel()
    order = torch.argsort(ratio.reshape(-1), descending=True)[:report_top]
    # Broadcast the bound to the output's shape before indexing it. A 0-d or partially-broadcast
    # bound is legal (a uniform absolute tolerance is a scalar, and `frac * ref.std()` is 0-d), but
    # `safe[c]` with a full coordinate then raises IndexError -- ON THE FAILURE PATH ONLY, so every
    # passing test looks fine and the one run that needed the diagnostic gets a traceback about
    # indexing instead. Measured: a 0-d bound turned a real mismatch into
    # "IndexError: too many indices for tensor of dimension 0".
    per_element = (
        torch.broadcast_to(safe, a.shape) if torch.is_tensor(safe) else torch.full_like(a, safe)
    )
    lines = []
    for flat in order.tolist():
        c = torch.unravel_index(torch.tensor(flat), a.shape)
        c = tuple(int(i) for i in c)
        # The coordinate on the split axis is slice-local; `offset` puts it back in the whole
        # tensor. A slice-local index would point at the wrong row, which is worse than none.
        reported = c[:offset_dim] + (c[offset_dim] + offset,) + c[offset_dim + 1 :] if c else c
        lines.append(
            f"    {reported}: actual={a[c].item():+.6g} ref={ref[c].item():+.6g} "
            f"err={err[c].item():.3g} bound={per_element[c].item():.3g} "
            f"ratio={ratio[c].item():.3f}"
        )
    raise AssertionError(
        f"{what}: {n}/{total} elements ({100.0 * n / total:.3f}%) exceed their per-element bound; "
        f"worst ratio {worst:.3f}. {_describe('actual', actual)}\n" + "\n".join(lines)
    )


def assert_gemm_exact(D, A, B, what="GEMM"):
    """Assert ``D == A @ B^T`` **exactly**, for integer operands within the exactness window.

    The gate the module docstring derives. Only valid when the operands are integers satisfying
    ``K * P * Q < 2**24``; this function *verifies* that rather than trusting the caller, because
    a silently-violated precondition turns a strong gate into a flaky one.

    Args:
        D: The kernel's output, ``(..., M, N)``. Its dtype must represent the exact result: with
            integer operands the products are integers, so an fp16 D must hold results within
            ±2048. Checked.
        A: ``(..., M, K)``, integer-valued.
        B: ``(..., N, K)``, integer-valued.
        what: Short label for the message.

    Returns:
        None.

    Raises:
        AssertionError: If the operands are not integer-valued, if the exactness window is
            violated, if the exact result does not fit ``D``'s dtype, or if any output element
            differs from the exact reference.
    """
    record_assertion("assert_gemm_exact")
    for name, t in (("A", A), ("B", B)):
        frac = (t.double() - t.double().round()).abs().max()
        assert float(frac) == 0.0, (
            f"{what}: {name} is not integer-valued (max fractional part {float(frac):.3g}); "
            f"assert_gemm_exact's argument only holds for integers. Use assert_gemm_close."
        )
    K = A.shape[-1]
    P = float(A.double().abs().max())
    Q = float(B.double().abs().max())
    assert K * P * Q < _FP32_EXACT_INT_MAX, (
        f"{what}: K*P*Q = {K}*{P:.0f}*{Q:.0f} = {K * P * Q:.0f} reaches fp32's exact-integer "
        f"ceiling {_FP32_EXACT_INT_MAX}; partial sums may round and the result is NOT provably "
        f"exact. Shrink the operand bound (see max_exact_operand) or use assert_gemm_close."
    )
    ref = gemm_reference(A, B)
    fits = exact_int_bound(D.dtype)
    biggest = float(ref.abs().max())
    assert biggest <= fits, (
        f"{what}: the exact result reaches {biggest:.0f}, past what {D.dtype} represents exactly "
        f"({fits}); D itself would round and the comparison would fail for the wrong reason. "
        f"Shrink the operand bound."
    )
    _check_finite(what, D)
    assert_bitwise(D.double(), ref, what=f"{what} (integer-exact)")


def assert_gemm_close(D, A, B, what="GEMM", scale=1.0):
    """Assert ``D ~= A @ B^T`` against the **per-element** analytic bound, for random operands.

    The everyday gate: use it wherever the operands cannot be integers (a realistic distribution, a
    shape whose K is too large for the exactness window). Strictly weaker than
    :func:`assert_gemm_exact`, so prefer that one where it applies.

    Args:
        D: The kernel's output, ``(..., M, N)``.
        A: ``(..., M, K)`` operand, any float dtype.
        B: ``(..., N, K)`` operand.
        what: Short label for the message.
        scale: Multiplier on the analytic bound. **Leave at 1.0 unless there is a stated reason**;
            it exists so a kernel with a documented extra rounding step (a split-K reduction, an
            fp8 rescale) can widen the bound explicitly rather than by loosening a tolerance
            constant nobody can re-derive.

    Returns:
        The worst observed ``|err| / bound`` ratio, for the caller to report.

    Raises:
        AssertionError: On non-finite output, a degenerate (all-zero) reference, or any element
            outside its bound.
    """
    record_assertion("assert_gemm_close")
    ref = gemm_reference(A, B)
    assert float(ref.abs().max()) > 0.0, (
        f"{what}: the reference is all zeros, so every comparison passes vacuously. Check the "
        f"operand construction -- this is what a silently-empty test looks like."
    )
    bound = gemm_error_bound(A, B, D.dtype) * scale
    return assert_elementwise(D, ref, bound, what=what)


def assert_written(run, shape, dtype, device, what="output", alloc=None):
    """Assert the kernel writes **every** element of its output.

    Runs ``run`` twice into buffers pre-filled with two different values, and requires the results
    to agree bitwise. Any element the kernel does not store keeps its pre-fill and the two runs
    disagree there -- so this catches a skipped tail tile or an under-predicated store, which a
    value comparison against a zeroed buffer misses whenever the reference is itself near zero.

    Chosen over a magic poison value deliberately: a poison must be a value the kernel cannot
    legitimately produce, and no such value exists in general for a float output.

    Args:
        run: A callable taking the output tensor and filling it in place. Must be deterministic --
            a kernel with a genuinely non-deterministic reduction order would fail this for a
            reason that is not the one being tested.
        shape: Output shape to allocate.
        dtype: Output dtype.
        device: Where to allocate.
        what: Short label for the message.
        alloc: Optional ``alloc(shape, dtype, device, value) -> tensor`` used instead of
            ``torch.full``. Needed whenever the output's row pitch must be padded to meet a
            hardware alignment floor -- a contiguous allocation at an off-grid trailing extent is
            refused by the TMA descriptor, so the default would fail for exactly the ragged shapes
            this gate is most useful at.

    Returns:
        The output tensor from the first run, so the caller can then check its values.

    Raises:
        AssertionError: If the two runs disagree anywhere, naming the count and first coordinates
            -- which are exactly the elements that were never stored.
    """
    record_assertion("assert_written")
    if alloc is None:
        d1 = torch.full(tuple(shape), 1.0, dtype=dtype, device=device)
        d2 = torch.full(tuple(shape), -2.0, dtype=dtype, device=device)
    else:
        d1 = alloc(shape, dtype, device, 1.0)
        d2 = alloc(shape, dtype, device, -2.0)
    run(d1)
    run(d2)
    if torch.equal(d1, d2):
        return d1
    diff = d1 != d2
    n = int(diff.sum())
    coords = [tuple(int(i) for i in c) for c in torch.nonzero(diff)[:3]]
    raise AssertionError(
        f"{what}: {n}/{d1.numel()} elements ({100.0 * n / d1.numel():.3f}%) were never written -- "
        f"they still hold the two different pre-fill values. First at {coords}. "
        f"This is a skipped tile or an over-tight store predicate, not a numeric error."
    )
