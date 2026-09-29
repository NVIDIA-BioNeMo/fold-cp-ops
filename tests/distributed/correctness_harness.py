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
"""Per-element and per-row error DISTRIBUTION of a distributed output -- a diagnostic, not a gate.

Purpose
    After a distributed store fails, the useful question is not "how wrong" but "wrong WHERE". This
    reports the error's shape: percentiles across elements, a per-token-row breakdown, and the
    ``(b, i, j)`` of the worst row. That is what distinguishes a uniformly-noisy result from one
    perfect row-collapse, and the two have entirely different causes.

**It reports; it does not decide.** The pass/fail bar is
`fold_cp_ops.testing.numerics`'s element-wise comparison, and nothing here should be asserted
against as a substitute. The distinction matters because this module's headline field, ``rel_l2``
(``||got-ref|| / ||ref||``), is exactly the pooled scalar that rule exists to forbid -- and the
upstream's own docstring for this file explains why, before using it as the gate anyway:

    *"a global rel_L2 AVERAGES error over all B*N*N*D elements, so a localized corruption -- e.g.
    the row-0 collapse where ONE token row is totally wrong but the other N-1 are perfect -- is
    hidden (1/N of the mass barely moves the norm)."*

    The upstream's answer was to pair the scalar with a per-ROW outlier count, which is a real
    improvement and is kept here. This tree's answer is to judge per ELEMENT, which subsumes it: a
    single corrupted element fails at its own index with both values, without needing to dominate a
    norm or to be one of enough elements in a row to move that row's L2. So ``rel_l2`` survives as a
    number to PRINT, ``n_outlier_rows`` survives as a cheap localization signal worth asserting
    alongside the element-wise bound, and neither is the bar.

Scope
    A deliberately minimal extraction, grown on demand. The upstream module is 533 lines and ten
    public symbols; the rule here is "restore a symbol when something calls it, not before", because
    a large module of untested helpers carried "because it is verbatim" is the failure mode the
    docstring rule exists to stop.

    Present: the error histogram (the original resident), and -- added when the A2A-fused TriMul
    workflow's parity tests began calling them -- :func:`make_weights`, :func:`global_oracle` and
    :func:`oracle_row_tile`. Still absent, still uncalled: the DTensor reference comparison and the
    streamed pass/fail wrappers, whose job (deciding) belongs to
    `fold_cp_ops.testing.numerics` here anyway.

Note
    ``from __future__ import annotations`` is present and is safe HERE, unlike in the kernel-adjacent
    modules where it is forbidden project-wide: it stringizes parameter annotations, which turns a
    ``cutlass.Constexpr`` parameter dynamic. This module imports torch and nothing from the CuTe DSL,
    so it has no annotation the DSL will ever read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ErrorHistogram:
    """Per-element and per-token-row error distribution of ``got`` vs ``ref``.

    All errors are absolute ``|got - ref|`` in fp32. The per-row error is the L2 norm over the
    feature axis (last dim) for each token, so a "row" is one token's D-vector and a dropped or
    collapsed token row shows a per-row error orders of magnitude above the median.

    Attributes:
        rel_l2: Global relative L2 ``||got-ref|| / ||ref||``. **A pooled scalar -- print it, do not
            gate on it.** See the module docstring.
        max_abs: Max absolute element error.
        mean_abs: Mean absolute element error.
        pct: Absolute-error percentiles, keys ``p50 p90 p99 p99.9 p100``.
        n_over_thresh: Count of elements whose absolute error exceeds ``abs_thresh``.
        frac_over_thresh: That count as a fraction of all elements.
        abs_thresh: The element threshold used -- see :func:`compute_error_histogram`.
        n_rows: Token-row count.
        row_med: Median per-row L2 error.
        row_max: Max per-row L2 error.
        row_outlier_ratio: ``row_max / row_med`` -- the localized-corruption fingerprint. A clean
            result has all rows in-distribution (ratio O(1-10)); a collapsed row spikes this by
            orders of magnitude.
        n_outlier_rows: Rows whose per-row error exceeds BOTH ``outlier_ratio x`` the median AND an
            absolute floor. Worth asserting ``== 0`` alongside the element-wise bound: it is a
            count of localized failures rather than an average over them.
        worst_row_index: ``(b, i, j)`` of the worst row -- where the error concentrates. A row-0
            collapse lands here at ``i == 0`` or ``j == 0``.
    """

    rel_l2: float
    max_abs: float
    mean_abs: float
    pct: dict
    n_over_thresh: int
    frac_over_thresh: float
    abs_thresh: float
    n_rows: int
    row_med: float
    row_max: float
    row_outlier_ratio: float
    n_outlier_rows: int
    worst_row_index: tuple

    def summary(self) -> str:
        """One line carrying every field -- the string a failing test prints alongside its bound."""
        p = self.pct
        return (
            f"rel_L2={self.rel_l2:.3e} | abs max={self.max_abs:.3e} mean={self.mean_abs:.3e} "
            f"p50={p['p50']:.2e} p90={p['p90']:.2e} p99={p['p99']:.2e} p99.9={p['p99.9']:.2e} | "
            f"elems>{self.abs_thresh:.1e}: {self.n_over_thresh} ({self.frac_over_thresh:.2%}) | "
            f"rows={self.n_rows} row_med={self.row_med:.3e} row_max={self.row_max:.3e} "
            f"ratio={self.row_outlier_ratio:.1f}x outlier_rows={self.n_outlier_rows} "
            f"worst@{self.worst_row_index}"
        )


def compute_error_histogram(
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    abs_thresh: Optional[float] = None,
    outlier_ratio: float = 50.0,
    outlier_abs_floor: Optional[float] = None,
) -> ErrorHistogram:
    """Per-element and per-row error distribution of ``got`` vs ``ref``, computed in fp32.

    Semantics
        Both tensors are detached and cast to fp32 before any arithmetic, so a bf16 pair is
        compared at fp32 precision rather than at bf16's. The per-row reduction is
        ``flatten(0, 2).norm(dim=-1)``: the LAST axis is treated as the feature axis and the first
        three as the row index. For the ``(B, N, N, D)`` shape this reads exactly as documented; for
        the 5-D recv ``(cp, Dloc, B, N_loc, N)`` the same call still produces a meaningful
        localization -- rows become ``(cp*Dloc*B, N_loc)`` and the feature axis is the token-j
        extent -- but ``worst_row_index`` is then an index into that flattening, not a ``(b, i, j)``.

        ``torch.quantile`` caps its input size, so above 16M elements the percentiles are computed
        on a deterministic stride-subsample rather than on every element. Percentiles are a
        diagnostic, so a subsample is acceptable there in a way it would never be for a bound.

    Args:
        got: The distributed output. Any dtype; compared in fp32.
        ref: The reference. **Must have exactly ``got``'s shape** -- asserted, because a broadcast
            would silently compare a tensor against a stretched view of a smaller one and report a
            plausible error distribution for a comparison nobody asked for.
        abs_thresh: Element absolute-error threshold for the count-over-threshold metric. Default
            scales with the reference magnitude, ``2e-2 * ref.abs().mean()``, so the count stays
            meaningful across sizes rather than measuring how large the tensor is.
        outlier_ratio: A row counts as localized corruption when its per-row L2 error exceeds this
            multiple of the MEDIAN per-row error (and clears the absolute floor). Default 50, well
            above clean bf16 row-to-row variation (O(1-10) in practice) and far below a collapsed
            row, which is orders of magnitude out. Must be positive.
        outlier_abs_floor: Absolute per-row error floor below which a row is never flagged, guarding
            against a tiny-median false positive when the whole output is near-exact. Defaults to
            ``2e-2 x`` the median per-row reference magnitude.

    Returns:
        The :class:`ErrorHistogram`. Every field is a plain Python number, so it is safe to print,
        format, or carry across a rank boundary.

    Raises:
        AssertionError: If ``got`` and ``ref`` have different shapes.
    """
    got = got.detach().float()
    ref = ref.detach().float()
    assert got.shape == ref.shape, f"shape mismatch {got.shape} vs {ref.shape}"

    # ONE difference tensor, reused, and turned into magnitudes IN PLACE. `got - ref` used to be
    # evaluated three separate times -- here, for `rel_l2`, and again for `row_err` below -- so a
    # comparison of a large recv held four fp32 copies of the same 2.1e9-element tensor at its peak.
    # Measured at cp=16 N_token=2048 D=256 on 2x8 H100: every rank died with
    # `torch.OutOfMemoryError: Tried to allocate 16.00 GiB` with 44.03 GiB already live, and the
    # ours-vs-main dump the run existed to produce was never written. The arithmetic is unchanged --
    # same values, same order -- only the number of simultaneously live temporaries is.
    diff = got - ref
    ref_norm = ref.norm().item() + 1e-30
    rel_l2 = diff.norm().item() / ref_norm
    # Both row reductions read the SIGNED difference, so they must run before `abs_()` below.
    row_err = diff.flatten(0, 2).norm(dim=-1)
    ref_row = ref.flatten(0, 2).norm(dim=-1)  # per-row ref magnitude
    err = diff.abs_()  # in place: `err` aliases `diff`'s storage, and `diff` is not read again
    max_abs = err.max().item()
    mean_abs = err.mean().item()

    flat = err.flatten()
    qs = torch.tensor([0.50, 0.90, 0.99, 0.999, 1.0], device=flat.device, dtype=flat.dtype)
    pv = (
        torch.quantile(flat, qs)
        if flat.numel() <= 16_000_000
        else torch.tensor(
            # torch.quantile caps input size; subsample deterministically for huge tensors.
            [flat[:: max(1, flat.numel() // 8_000_000)].quantile(q).item() for q in qs.tolist()],
            device=flat.device,
        )
    )
    pct = {
        "p50": pv[0].item(),
        "p90": pv[1].item(),
        "p99": pv[2].item(),
        "p99.9": pv[3].item(),
        "p100": pv[4].item(),
    }

    if abs_thresh is None:
        abs_thresh = 2e-2 * (ref.abs().mean().item() + 1e-30)
    # Count over the threshold WITHOUT a full-size boolean mask. `(err > t).sum()` allocates one
    # byte per element beside `err`, which is the single largest request in this function at recv
    # scale. A count is additive over a partition, so summing over blocks is EXACT -- not a sample,
    # not an approximation -- and caps the extra allocation at one block regardless of input size.
    _blk = 1 << 26  # 64Mi elements: 64 MiB of mask, small beside any tensor that needs chunking
    _thr = float(abs_thresh)  # bound once: the default above is assigned inside an `is None` branch
    n_over = 0
    for _s in range(0, flat.numel(), _blk):
        n_over += int((flat[_s : _s + _blk] > _thr).sum().item())
    frac_over = n_over / err.numel()

    n_rows = row_err.numel()
    row_med = row_err.median().item()
    row_max = row_err.max().item()
    row_outlier_ratio = row_max / (row_med + 1e-30)
    if outlier_abs_floor is None:
        outlier_abs_floor = 2e-2 * (ref_row.median().item() + 1e-30)
    # A row is a localized-corruption outlier if BOTH: error >> median AND above the absolute floor
    # (so a uniformly-near-exact output never trips).
    outlier_mask = (row_err > outlier_ratio * (row_med + 1e-30)) & (row_err > outlier_abs_floor)
    n_outlier_rows = int(outlier_mask.sum().item())
    worst_flat = int(row_err.argmax().item())
    _, N1, N2 = ref.shape[0], ref.shape[1], ref.shape[2]
    bi = worst_flat // (N1 * N2)
    ii = (worst_flat % (N1 * N2)) // N2
    jj = worst_flat % N2
    worst_row_index = (bi, ii, jj)

    return ErrorHistogram(
        rel_l2=rel_l2,
        max_abs=max_abs,
        mean_abs=mean_abs,
        pct=pct,
        n_over_thresh=n_over,
        frac_over_thresh=frac_over,
        abs_thresh=abs_thresh,
        n_rows=n_rows,
        row_med=row_med,
        row_max=row_max,
        row_outlier_ratio=row_outlier_ratio,
        n_outlier_rows=n_outlier_rows,
        worst_row_index=worst_row_index,
    )


def compute_error_histogram_blocked(
    blocks,
    *,
    abs_thresh: float,
    row_dims: tuple,
    outlier_ratio: float = 50.0,
    outlier_abs_floor: Optional[float] = None,
) -> ErrorHistogram:
    """Streaming :func:`compute_error_histogram` -- never materializes the whole reference.

    Purpose
        Compare an output against a reference that DOES NOT FIT beside it. The front recv holds
        ``2*Dloc x cp*B*N^2`` elements, which simplifies to ``2*D*B*N^2`` -- ``cp`` CANCELS, so
        sharding across more ranks does not shrink a rank's copy. Measured at ``cp=(2,2)``,
        ``N_token=5952``, ``D=256``: the reference alone asked for a single **33.79 GiB**
        allocation beside an equally large recv, and every mesh failed at exactly the same
        ``N_token`` -- 0/6 at 4096 and 0/6 at 5952, identically for cp 2, 4, 8, 16 and (2,2).

    Semantics
        Statistically IDENTICAL to :func:`compute_error_histogram`, not an approximation of it,
        with one documented exception. Every reported quantity is additive or associative over a
        partition of the ROW axis, so accumulating over blocks is exact:
        ``max`` of maxima, ``sum``/count for the mean, sums of squares for ``rel_l2``, a count for
        ``n_over_thresh``, and per-row norms that are already complete within a block.
        **The exception is ``pct``**: percentiles are not decomposable, so a deterministic
        stride-subsample is taken per block and quantiled at the end. The non-streaming function
        already subsamples above 16M elements for the same reason, and percentiles are a
        diagnostic there too -- never a bound.

    Args:
        blocks: Iterable of ``(got_block, ref_block)`` pairs, each ``(rows_in_block, n_features)``
            and mutually the same width, together partitioning the row axis IN ORDER. Any dtype;
            compared in fp32. **A block whose two halves differ in shape is a caller bug** and is
            asserted, because a broadcast would silently compare against a stretched view.
        abs_thresh: The element threshold, **required and not defaulted**. The non-streaming
            function derives it from ``ref.abs().mean()``, which a single pass over a stream cannot
            know before it has consumed the stream; a second pass would re-run the collectives that
            produce the blocks. The caller computes it from data it already holds and passes it in,
            so the count stays exact rather than becoming threshold-dependent on block order.
        row_dims: ``(B, N1, N2)`` used only to map the worst flat row index to ``(b, i, j)``. Wrong
            dims mislabel that ONE diagnostic; they do not affect any measured quantity.
        outlier_ratio: As in :func:`compute_error_histogram`. Must be positive.
        outlier_abs_floor: As in :func:`compute_error_histogram`; defaults to ``2e-2 x`` the median
            per-row reference magnitude, computed once over the accumulated rows.

    Returns:
        The :class:`ErrorHistogram`, with the same field meanings as the non-streaming function.

    Raises:
        AssertionError: If a block pair's shapes disagree, if no block was yielded at all (an empty
            stream would otherwise report a vacuous all-zero histogram that reads as a pass), or if
            block widths are inconsistent.
    """
    n_elem = 0
    sum_abs = 0.0
    sum_sq_diff = 0.0
    sum_sq_ref = 0.0
    max_abs = 0.0
    n_over = 0
    row_errs = []
    ref_rows = []
    samples = []
    width = None
    for got_b, ref_b in blocks:
        got_b = got_b.detach().float()
        ref_b = ref_b.detach().float()
        assert got_b.shape == ref_b.shape, f"block shape mismatch {got_b.shape} vs {ref_b.shape}"
        if width is None:
            width = got_b.shape[-1]
        assert got_b.shape[-1] == width, f"block width changed {got_b.shape[-1]} != {width}"
        diff = got_b - ref_b
        sum_sq_diff += float(diff.pow(2).sum().item())
        sum_sq_ref += float(ref_b.pow(2).sum().item())
        row_errs.append(diff.norm(dim=-1))
        ref_rows.append(ref_b.norm(dim=-1))
        err = diff.abs_()  # in place: `diff` is not read again (see the non-streaming twin)
        n_elem += err.numel()
        sum_abs += float(err.sum().item())
        max_abs = max(max_abs, float(err.max().item()))
        n_over += int((err > abs_thresh).sum().item())
        flat = err.reshape(-1)
        samples.append(flat[:: max(1, flat.numel() // 1_000_000)].clone())
    assert width is not None, "compute_error_histogram_blocked consumed an EMPTY stream"

    row_err = torch.cat(row_errs)
    ref_row = torch.cat(ref_rows)
    samp = torch.cat(samples)
    qs = torch.tensor([0.50, 0.90, 0.99, 0.999, 1.0], device=samp.device, dtype=samp.dtype)
    pv = (
        torch.quantile(samp, qs)
        if samp.numel() <= 16_000_000
        else torch.tensor(
            [samp[:: max(1, samp.numel() // 8_000_000)].quantile(q).item() for q in qs.tolist()],
            device=samp.device,
        )
    )
    pct = {
        "p50": pv[0].item(),
        "p90": pv[1].item(),
        "p99": pv[2].item(),
        "p99.9": pv[3].item(),
        "p100": pv[4].item(),
    }
    # p100 is the max of a SUBSAMPLE; `max_abs` is the true max over every element. Report the true
    # one in the field that a reader gates on.
    pct["p100"] = max(pct["p100"], max_abs)

    row_med = row_err.median().item()
    row_max = row_err.max().item()
    if outlier_abs_floor is None:
        outlier_abs_floor = 2e-2 * (ref_row.median().item() + 1e-30)
    outlier_mask = (row_err > outlier_ratio * (row_med + 1e-30)) & (row_err > outlier_abs_floor)
    worst_flat = int(row_err.argmax().item())
    _, N1, N2 = row_dims
    return ErrorHistogram(
        rel_l2=(sum_sq_diff**0.5) / ((sum_sq_ref**0.5) + 1e-30),
        max_abs=max_abs,
        mean_abs=sum_abs / n_elem,
        pct=pct,
        n_over_thresh=n_over,
        frac_over_thresh=n_over / n_elem,
        abs_thresh=abs_thresh,
        n_rows=row_err.numel(),
        row_med=row_med,
        row_max=row_max,
        row_outlier_ratio=row_max / (row_med + 1e-30),
        n_outlier_rows=int(outlier_mask.sum().item()),
        worst_row_index=(worst_flat // (N1 * N2), (worst_flat % (N1 * N2)) // N2, worst_flat % N2),
    )


# --------------------------------------------------------------------------- #
# Replicated weights + the fp32 global oracle. Added for the A2A-fused TriMul workflow's parity
# tests; see the Scope note above.
# --------------------------------------------------------------------------- #


def make_weights(D: int, *, in_bias: bool = False, out_bias: bool = False, seed: int, device) -> dict:
    """Deterministic replicated TriMul weights -- identical on every rank of a job.

    Purpose
        Give every rank the SAME weights without a broadcast. Under SPMD the ranks must agree
        bit-for-bit or the comparison against a global oracle is meaningless, and a per-rank
        ``torch.randn`` on the device would not agree even at a fixed seed.

    Semantics
        Generated on the CPU from an explicit `torch.Generator`, then moved -- so the values depend
        on the seed alone and not on the device, the CUDA RNG state, or the rank. The four
        projections are drawn at GLOROT/Xavier scale, ``std = sqrt(2/(fan_in+fan_out))``; the
        LayerNorm affine is ``gain = 1 + N(0, 0.1**2)`` and ``bias = N(0, 0.5**2)``. The projection
        biases, when asked for, are ``N(0, 0.4**2)`` -- see :func:`proj_b` for why not ``0.02``.

        **Xavier is here because it makes the BOUND mean something, not because it reduces noise.**
        It cannot reduce noise: bf16's relative error is scale-invariant and the einsum's fp32
        accumulation error over a length-``N_token`` contraction grows like ``sqrt(K)*eps``, and no
        choice of initializer moves either. What the previous fixed ``0.02`` did was leave the output
        so small that the flat ``atol`` term of `tolerance_bound` swamped the ``rtol`` term for
        **99.1%** of elements at D=128 -- an effective bar of 29% of the reference value -- and,
        because a GEMM contracting over ``D`` at a fixed operand scale produces output growing like
        ``sqrt(D)``, the same tolerance was ~1.6x more permissive at D=128 than at D=512. Measured
        ``bound/|ref|`` at the median, D = 128/256/512: **28.9 / 23.7 / 18.2%** at the old fixed scale
        versus **12.4 / 11.1 / 12.5%** here. Glorot's ``sqrt(2/(fan_in+fan_out))`` cancels the
        ``sqrt(D)`` growth exactly, which is what flattens that row.

        **The LayerNorm affine is deliberately OFF the identity, and that is the point.** It used to
        be ``gain = ones(D)`` / ``bias = zeros(D)`` -- the multiplicative and additive identities --
        which made the term it exists to test invisible: zeroing both LN biases and recomputing the
        fp32 oracle moved the answer by exactly ``0.0000``, so a fused kernel that never read
        ``norm_in_b``/``norm_out_b`` at all was bit-identical to a correct one. Measured at
        ``B=1, N=256, D=128``: the fraction of output elements where dropping the bias exceeds
        `tolerance_bound(ref, 2e-2, 6e-2)` goes from **0.0%** at zeros to **86.7%** at
        ``N(0, 0.5**2)``. The gain matters for the same reason and one more: ``alg_fold`` folds it
        into the GEMM weight on the host, so at ``gain == 1`` that fold is a no-op copy and a defect
        in it cannot be seen. `tests/distributed/test_correctness_harness.py` pins this property so
        it cannot rot back.

        The 0.5 scale is chosen, not arbitrary: after the normalize the signal is unit-variance, so
        a bias at ``N(0, s**2)`` owns ``s**2 / (1 + s**2)`` of the LN output variance -- 20% at
        s=0.5. Much smaller is undetectable against a 12-29% tolerance; much larger is WORSE, because
        a bias that dominates makes every token's activation nearly the same constant vector, which
        both sides compute identically and which drives the einsum toward rank-1.

        **The affine draws come from their own generator stream** (`seed + 7919`), so adding them
        left every projection weight bit-identical to what this function returned before. That keeps
        the change one-variable: a rung that goes red after this can only be the LN affine.

    Args:
        D: feature width. ``p_in_w`` / ``g_in_w`` are ``(2D, D)`` (the dual's two halves) and
            ``p_out_w`` / ``g_out_w`` are ``(D, D)``.
        in_bias: add ``p_in_b`` / ``g_in_b``, the FRONT projection biases. ``TriMulAutotuned``
            currently RAISES ``NotImplementedError`` on a non-None value here -- the front store does
            not fuse them yet -- so this exists to exercise that refusal, not to run a chain.
        out_bias: add ``p_out_b`` / ``g_out_b``, the BACK projection biases. These the workflow
            really does fuse, as ``bg`` / ``bp`` on `layernorm_dual_gated_gemm`.

            **These are two flags because they used to be one, and that is why the fused pair went
            untested.** A single ``bias=`` drove all four, so asking for the output biases also asked
            for the input ones and tripped the front refusal before anything ran -- making
            ``bias=True`` unusable and leaving ``p_out_b``/``g_out_b`` with no coverage at all
            despite being live, fused code. Splitting them is what makes the supported half
            reachable. Both default False, the wave-1 contract.
        seed: the generator seed. Must be the same value on every rank; a per-rank seed silently
            produces a per-rank model.
        device: destination device for every tensor.

    Returns:
        The ``w`` dict, with the same twelve keys `trimul_weights.W_KEYS` /
        `trimul_weights.W_PROJ_BIAS_KEYS` declare.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.02).to(device)

    def xavier(fan_out, fan_in):
        """A ``(fan_out, fan_in)`` projection at Glorot scale, ``std = sqrt(2/(fan_in+fan_out))``.

        Computed by hand rather than via `torch.nn.init.xavier_normal_` on purpose. That function
        DOES accept ``generator=`` on torch 2.11, but it is easy to call without one, and a
        global-RNG draw here silently produces a PER-RANK model -- which surfaces as a correctness
        failure that reads like a kernel bug. Drawing through the explicit `g` cannot do that.

        It also consumes exactly as many values from `g` as the fixed-scale draw it replaced, so
        every later draw in this function keeps its position in the stream.
        """
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return (torch.randn(fan_out, fan_in, generator=g, dtype=torch.float32) * std).to(device)

    # SEPARATE stream, drawn BEFORE the dict so `g`'s sequence is untouched -- the projections come
    # out bit-identical to what this function returned when the affine was ones/zeros. The offset is
    # a prime rather than `seed + 1` because callers already use `seed + 1` for the input tensor
    # (`test_trimul_autotuned.py`), and two Generators on the same seed yield the SAME numbers -- the
    # gain would have been the first D values of `x`.
    g_ln = torch.Generator(device="cpu").manual_seed(seed + 7919)

    def ln_affine():
        """One LayerNorm's ``(gain, bias)``, both fp32 and both OFF the identity.

        fp32 is required, not incidental: `layernorm_dual_gated_gemm` refuses a non-fp32
        `norm_weight`, and `trimul_autotuned.py` assigns `self._norm_out_w = w["norm_out_w"]` with no
        `.float()`, so this function is where the dtype is decided.
        """
        gain = (1.0 + torch.randn(D, generator=g_ln, dtype=torch.float32) * 0.1).to(device)
        bias = (torch.randn(D, generator=g_ln, dtype=torch.float32) * 0.5).to(device)
        return gain, bias

    def proj_b(n):
        """A projection bias at 0.4, scaled to the projection OUTPUT rather than to its weights.

        Not `rn`'s 0.02, and the difference is measured. Under Xavier the pre-activation has std
        ``sqrt(fan_in) * sqrt(2/(fan_in+fan_out)) ~= sqrt(2/3) ~= 0.82``, D-independently, so a 0.02
        bias is a 2.5% perturbation of the thing it is meant to shift -- too small to detect.
        Measured at `rn`'s scale: dropping the bias entirely moved only **15.8%** (front pair) and
        **0.8%** (back pair) of the output past the e2e bound, so a kernel that ignored it passed.

        The BACK pair is the weaker of the two and needs the headroom most: ``g_out_b`` shifts a
        sigmoid pre-activation, and the sigmoid's derivative is at most 0.25, so its effect is damped
        ~4x before the comparison ever sees it. Same 20%-of-variance rule as the LayerNorm bias,
        applied to a signal of std 0.82 rather than 1.
        """
        return (torch.randn(n, generator=g, dtype=torch.float32) * 0.4).to(device)

    norm_in_w, norm_in_b = ln_affine()
    norm_out_w, norm_out_b = ln_affine()

    return dict(
        norm_in_w=norm_in_w, norm_in_b=norm_in_b,
        p_in_w=xavier(2 * D, D), g_in_w=xavier(2 * D, D),
        norm_out_w=norm_out_w, norm_out_b=norm_out_b,
        p_out_w=xavier(D, D), g_out_w=xavier(D, D),
        p_in_b=proj_b(2 * D) if in_bias else None, g_in_b=proj_b(2 * D) if in_bias else None,
        p_out_b=proj_b(D) if out_bias else None, g_out_b=proj_b(D) if out_bias else None,
    )


def oracle_row_tile(N: int, D: int):
    """Adaptive ``row_tile`` for the chunked oracle, or ``None`` meaning "use the exact reference".

    Purpose
        Keep one answer to "how big a tile" so every suite picks the same one instead of
        re-deriving it, and so a shape that fits stays on the byte-identical exact path.

    Semantics
        Bounds one operand tile to ~100 Mi elements, so the transient projection intermediates stay
        a few GB REGARDLESS of ``D``. A fixed 512 left 15-22 GB at N=4096, D=384 and still OOM'd,
        which is why the tile scales with ``N*D`` rather than being a constant.

    Args:
        N: token extent. ``N <= 512`` returns ``None`` -- the exact reference fits, and it is the
            one the chunked variant is validated against.
        D: feature width.

    Returns:
        A tile in ``[64, 512]``, or ``None``.
    """
    return max(64, min(512, (100 << 20) // (int(N) * int(D)))) if int(N) > 512 else None


def global_oracle(x_global: torch.Tensor, weights: dict, *, direction: str,
                  mask_global: Optional[torch.Tensor], eps: float,
                  row_tile: Optional[int] = None) -> torch.Tensor:
    """fp32 ground truth: the TriMul reference on the FULL (un-sharded) global input.

    Purpose
        The thing a distributed result is compared against. Deliberately computed from the global
        input rather than by assembling per-rank results, so a resharding bug cannot cancel itself.

    Semantics
        ``row_tile=None`` runs the exact batched reference. A non-None value dispatches to the
        chunked variant, which tiles the (i, j) OUTPUT grid while keeping the k-contraction WHOLE --
        so the reduction order, and therefore the fp32 result, is unchanged. Measured on this tree:
        rel_L2 ``0.000e+00`` against the exact reference for both directions at row tiles that do
        and do not divide N.

        Both entries live in `fold_cp_ops.workflows.trimul_autotune`, the SHIPPED package -- not
        under ``benchmark/`` where the upstream kept them. An oracle a ``pip install``ed user cannot
        import is one they cannot check against.

    Args:
        x_global: ``(B, N, N, D)`` full input. Must be the same tensor on every rank.
        weights: a `make_weights` dict.
        direction: ``"outgoing"`` or ``"incoming"``.
        mask_global: ``(B, N, N)`` or ``None``.
        eps: LayerNorm epsilon; must match the value the kernel ran with, or the comparison measures
            the epsilon rather than the kernel.
        row_tile: see `oracle_row_tile`. ``None`` is the exact path.

    Returns:
        ``(B, N, N, D)`` fp32.
    """
    if row_tile is not None:
        from fold_cp_ops.workflows.trimul_autotune import trimul_ref_chunked

        return trimul_ref_chunked(
            x_global, direction, mask_global,
            weights["norm_in_w"], weights["norm_in_b"], weights["p_in_w"], weights["g_in_w"],
            weights["norm_out_w"], weights["norm_out_b"], weights["p_out_w"], weights["g_out_w"],
            weights.get("p_in_b"), weights.get("g_in_b"),
            weights.get("p_out_b"), weights.get("g_out_b"), eps,
            int(row_tile), None,
        )
    from fold_cp_ops.workflows.trimul_autotune import trimul_ref

    return trimul_ref(
        x_global, direction, mask_global,
        weights["norm_in_w"], weights["norm_in_b"], weights["p_in_w"], weights["g_in_w"],
        weights["norm_out_w"], weights["norm_out_b"], weights["p_out_w"], weights["g_out_w"],
        weights.get("p_in_b"), weights.get("g_in_b"),
        weights.get("p_out_b"), weights.get("g_out_b"), eps,
    )
