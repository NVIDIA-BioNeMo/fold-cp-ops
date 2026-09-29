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

"""The instruction-emitting half of the fused-LayerNorm prologue: reduce, finalize, normalize.

Four free functions, in the order the mainloop calls them:

``stage_ln_affine``    stage the ``(K,)`` fp32 gain/bias from global into shared, once per CTA
``accumulate_row_stats``  add one staged A tile's per-row ``Sum x`` / ``Sum x^2`` into registers
``finalize_row_stats``    combine the partials into ``(mu, rstd)`` and publish them to shared
``normalize_tile``     rewrite one staged A tile in place as ``(x - mu) * rstd * gain + bias``

They are FREE FUNCTIONS, not methods on a mixin, for two reasons. The first is testability: each can
be driven from a small standalone kernel with no GEMM around it, which is what
``tests/_internal/test_ln_prologue.py`` does. The second is that they are shared by two callers with
different class hierarchies -- the fused dual-gated projection and, later, its all-to-all-fused
distributed sibling -- and a mixin would have forced both onto one MRO.

**The arithmetic is the contract, and the ORDER within it is part of the contract.** The normalize
is written ``(x - mu) * rstd * gain`` and then ``+ bias``, not ``x * (rstd*gain) + (bias - mu*rstd*gain)``
which is the same function and a different bit pattern; the variance is ``E[x^2] - mu^2`` in fp32,
not a two-pass centred sum. Both choices reproduce the upstream physical-fusion kernel exactly.
Changing either is a numerical change, so it is a change to this module's observable behaviour even
though no signature moves.

**What this module does NOT do.** It never touches a global output, never issues an MMA, and never
allocates: every buffer arrives as an argument. That is what lets the same four calls sit inside a
one-pass mainloop, a two-pass streaming mainloop, or a cluster-distributed one without knowing
which.
"""

import operator
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

__all__ = [
    "RS_QUAD_LANES",
    "RS_ROW_STRIDE",
    "accumulate_row_stats",
    "finalize_row_stats",
    "make_row_stat_partials",
    "normalize_tile",
    "rs_accumulate_row_stats",
    "rs_owned_rows",
    "stage_ln_affine",
]

#: Lanes of a warp that split ONE row's K share in the WGMMA operand-A register fragment. The
#: cross-lane step that completes a row's K-sum is a `warp_reduction` over exactly this many lanes.
RS_QUAD_LANES = 4

#: Distance in M-rows between the two rows one thread owns: it holds ``base`` and ``base + 8``.
RS_ROW_STRIDE = 8


def rs_owned_rows(tidx):
    """Which two M-rows this thread owns in the WGMMA operand-A REGISTER fragment, and whether it
    is the lane that writes them.

    Purpose
        The register-source LayerNorm reduces statistics out of the operand-A fragment instead of
        re-reading the staged tile, so it needs to know which output rows a thread's registers
        correspond to. That mapping is the ONE piece of the register path that is shared between
        the fusions rather than forked, because it is a pure function of the thread index with no
        pipeline state -- and because a second, drifting copy of it would produce wrong statistics
        silently rather than failing.

    Semantics
        **The map is EMPIRICALLY DERIVED, not documented by the hardware.** It was obtained by
        encoding the ``(m, k)`` coordinate into a bf16 staged tile and reading back which register
        slot each value landed in. It is a closed form of the fragment for a **128-row tile split
        across two warpgroups**::

            base = (t // 128) * 64 + ((t % 128) // 32) * 16 + ((t % 32) // 4)

        Read left to right: ``t // 128`` selects the warpgroup (rows 0..63 or 64..127, disjoint, so
        no row is written twice), ``(t % 128) // 32`` the warp band within it, and ``(t % 32) // 4``
        the row within the band. Over 256 threads it enumerates 0..127 exactly once and NOTHING
        else -- which is precisely why every caller must gate on that tile. At ``tile_M = 256`` a
        thread owns four rows and this form addresses only the first 96; at 64 or 192 the warpgroup
        split changes underneath it. In both cases the kernel answers plausibly and WRONGLY, which
        is why the callers refuse the tile at the front door rather than trusting this function.

        The four lanes of a quad (``t % 4`` in 0..3) hold DISJOINT K of the SAME two rows, so each
        row's K-sum is completed by a `RS_QUAD_LANES`-wide warp reduction and written once, by the
        owner lane.

        Pure integer arithmetic, so it evaluates identically on a Python ``int`` and on a DSL
        integer -- which is what makes it testable without a GPU.

    Args:
        tidx: This thread's index within the cooperating group, 0..255 cooperatively. Must be the
            GLOBAL index across both warpgroups, not a warpgroup-local one: the first term selects
            the warpgroup, so a reduced index maps both warpgroups onto rows 0..63 and half the
            tile is never written.

    Returns:
        ``(base, is_owner)`` -- the first of the two owned rows (the second is
        ``base + RS_ROW_STRIDE``), and whether this thread is the lane that writes them.
    """
    return (
        (tidx // 128) * 64 + ((tidx % 128) // 32) * 16 + ((tidx % 32) // RS_QUAD_LANES),
        (tidx % RS_QUAD_LANES) == 0,
    )


@cute.jit
def rs_accumulate_row_stats(tCrA, row_sum, row_sqsum):
    """Add one k-tile's per-row ``Sum x`` / ``Sum x^2`` from the operand-A REGISTER fragment.

    Purpose
        The register-source path's reduction. It replaces the shared-memory re-read that
        `accumulate_row_stats` performs with a reduction over registers the WGMMA has already been
        given, which is the entire reason that path exists for an MN-major activation.

    Semantics
        **The reduction profile encodes the SAME empirically-derived fragment layout as
        `rs_owned_rows`, which is why the two live together.** The operand-A value mode is
        ``(k_lo, M, k_hi)``: sub-mode 1 is the TWO owned M-rows, sub-modes 0 and 2 are halves of
        this thread's K share, and the trailing ``CPY_N`` mode is more K. The profile
        ``((0, None, 1), None, 1)`` keeps ONLY that M sub-mode and sums everything else, leaving one
        partial per owned row.

        A profile that kept the wrong sub-mode would sum across ROWS instead of within them and
        produce a LayerNorm that still looks plausible -- the same silent-wrong failure mode that
        makes the row map worth sharing rather than copying. Forking one and sharing the other
        would put two halves of a single layout fact in two places.

        The four quad lanes hold DISJOINT K of the same two rows, so this leaves a per-lane partial;
        the cross-lane completion is a `RS_QUAD_LANES`-wide warp reduction and belongs to the
        caller's finalize, which is forked because it sequences a barrier.

        Sequences nothing -- no wait, no arrive, no sweep -- which is what makes it shareable across
        fusions whose sweep counts differ.

    Args:
        tCrA: This k-tile's operand-A register fragment, ALREADY filled by the s2r copy. Read only.
            Must be the fragment of a 128-row tile across two warpgroups, the layout the profile
            above is a closed form of; any other tile reduces the wrong sub-modes.
        row_sum: This thread's fp32 ``Sum x`` partials, one per owned row. Updated IN PLACE and
            also returned, so a caller's unrolled k-loop can rebind them.
        row_sqsum: Likewise for ``Sum x^2``.

    Returns:
        ``(row_sum, row_sqsum)`` -- the same two fragments.
    """
    x = tCrA.load().to(Float32)
    prof = ((0, None, 1), None, 1)  # keep ONLY the M sub-mode; reduce k_lo, k_hi, MMA_M, CPY_N
    tile_sum = x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=prof)
    tile_sqsum = (x * x).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=prof)
    for m in cutlass.range_constexpr(cute.size(row_sum)):
        row_sum[m] = row_sum[m] + tile_sum[m]
        row_sqsum[m] = row_sqsum[m] + tile_sqsum[m]
    return row_sum, row_sqsum


def make_row_stat_partials(thr_copy, sA_slice):
    """Allocate this thread's zeroed ``Sum x`` / ``Sum x^2`` register partials.

    Purpose
        The reduction runs across several k-tiles, so the partials must be created once OUTSIDE the
        k-loop and carried through it. Deriving their length here -- from the same partition the
        reduction will use -- is what stops a hand-written count from disagreeing with the tiling.

    Semantics
        The length is the partition's ``CPY_M`` mode: the number of ROWS this thread owns. It is a
        trace-time constant, so the returned fragments are registers, not local memory.

        A plain ``def``, not ``@cute.jit``: it returns two tensors, and the DSL flattens a jit
        function's return into MLIR values. The consequence is that it does not get the DSL
        preprocessor, so the zeroing loop below is a plain Python ``range`` over a compile-time
        constant -- which unrolls identically to ``cutlass.range_constexpr`` and, unlike it, does
        not raise "should be preprocessed by preprocessor" here.

    Args:
        thr_copy: This thread's slice of the statistics tiled copy, from
            ``compile_time.ln_prologue_layout.stats_tiled_copy``. Must be the SAME partition the
            reduction and the normalize use, or a thread accumulates one set of rows and rescales
            another.
        sA_slice: One staged ``(tile_M, tile_K)`` A tile, used only for its SHAPE. Any stage will
            do; the partition is stage-independent.

    Returns:
        ``(row_sum, row_sqsum)`` -- two fp32 register fragments of length ``CPY_M``, both zeroed.
    """
    cpy_m = cute.size(thr_copy.partition_S(sA_slice), mode=[1])
    row_sum = cute.make_rmem_tensor(cute.make_layout(cpy_m), Float32)
    row_sqsum = cute.make_rmem_tensor(cute.make_layout(cpy_m), Float32)
    for m in range(cpy_m):
        row_sum[m] = Float32(0.0)
        row_sqsum[m] = Float32(0.0)
    return row_sum, row_sqsum


@cute.jit
def stage_ln_affine(
    s_weight: cute.Tensor,
    s_bias: Optional[cute.Tensor],
    mWeight: cute.Tensor,
    mBias: Optional[cute.Tensor],
    gemm_k: cutlass.Constexpr[int],
    k_real: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    tidx: Int32,
    barrier,
) -> None:
    """Cooperatively copy the ``(K,)`` fp32 LayerNorm gain and bias from global into shared memory.

    Purpose
        Every normalizing thread reads the gain at its own columns, once per k-tile. Staging the
        vector once per CTA turns ``k_tiles * threads`` global reads into ``K`` of them, and the
        vector is small enough (``K`` fp32 values) that the shared-memory cost is negligible beside
        the A and B tiles.

    Semantics
        Strided over ``num_threads``: thread ``t`` writes indices ``t, t + num_threads, ...``. The
        tail ``[k_real, gemm_k)`` -- present only when the front door zero-extended a K that was not
        a multiple of the K tile -- is written as ZERO rather than left undefined. That is what
        makes the padding exact instead of merely out of range: with ``gain[k] == 0`` the normalize
        maps the TMA's zero-filled A column to ``(0 - mu) * rstd * 0 + 0 == 0``, which then
        contracts against the equally zero-filled B column. Leaving it undefined would put a NaN or
        a stale value into a real product.

        Ends with an async-shared fence and a barrier, so every thread sees the whole vector before
        the first normalize reads it. **Every thread of the cooperating group must reach this
        call** -- the barrier is collective, and partial participation hangs.

    Args:
        s_weight: The ``(gemm_k,)`` fp32 shared-memory destination for the gain. Must be at least
            ``gemm_k`` long: the normalize indexes it by the GLOBAL k of its tile, which runs to
            the PADDED extent, so a buffer sized to the true K would be read past its end.
        s_bias: The ``(gemm_k,)`` fp32 destination for the bias, or None when the kernel was built
            without one. Must be non-None exactly when ``mBias`` is.
        mWeight: The ``(k_real,)`` fp32 gain in global memory. fp32 is required, not converted:
            LayerNorm gain in 16 bits would lose more precision than the normalize itself.
        mBias: The ``(k_real,)`` fp32 bias in global memory, or None.
        gemm_k: The PADDED contraction extent -- the number of shared slots to fill.
        k_real: The TRUE feature width -- how many are copied from global. Must be ``<= gemm_k``;
            when equal, the zeroing branch is pruned entirely and the loop is the unpadded one.
        num_threads: Threads cooperating. Must be the same group that arrives at ``barrier``.
        tidx: This thread's index within that group, in ``[0, num_threads)``.
        barrier: The named barrier the cooperating group shares -- typically the kernel's epilogue
            barrier, since the same warpgroups do both jobs.

    Returns:
        None. Its effect is ``s_weight``/``s_bias``, visible to the whole group on return.
    """
    has_bias = const_expr(s_bias is not None)
    n_rounds = const_expr((gemm_k + num_threads - 1) // num_threads)
    for k in cutlass.range_constexpr(n_rounds):
        idx = k * num_threads + tidx
        if idx < k_real:
            s_weight[idx] = mWeight[idx]
            if const_expr(has_bias):
                s_bias[idx] = mBias[idx]
        elif const_expr(k_real < gemm_k):
            if idx < gemm_k:
                s_weight[idx] = Float32(0.0)
                if const_expr(has_bias):
                    s_bias[idx] = Float32(0.0)
    cute.arch.fence_view_async_shared()
    barrier.arrive_and_wait()


@cute.jit
def accumulate_row_stats(thr_copy, sA_slice: cute.Tensor, row_sum, row_sqsum):
    """Add one staged A tile's per-row ``Sum x`` and ``Sum x^2`` into this thread's fp32 partials.

    Purpose
        The first of the two passes over A. It reads only, so it may run while the WGMMA over the
        same staged tile is still in flight -- which is exactly how the one-pass caller uses it.

    Semantics
        Loads this thread's owned elements into registers, promotes to fp32, and reduces the
        partition's ``CPY`` (vector) and ``CPY_N`` (column) modes while KEEPING ``CPY_M`` (row).
        The per-lane result is a partial: the ``threads_per_row`` lanes cooperating on a row still
        hold disjoint pieces of it, and combining them is deliberately deferred to
        :func:`finalize_row_stats` so the butterfly runs once rather than once per k-tile.

        **THE CALLER MUST FENCE BETWEEN THIS AND RELEASING THE STAGE, and only one caller does.**
        The loads issued here are plain ``LDS``. A caller's ``wait_group(0)`` drains the WGMMA's
        reads of the same tile and NOT these, and the reduction above is pure register arithmetic,
        so the compiler sinks it past the release -- measurably does -- leaving only the load ISSUE
        ahead of the arrive. Release the stage there and the producer refills it while those loads
        are in flight, and they return the NEXT k-tile's activations: a silent wrong answer, not a
        crash. `cute.arch.fence_view_async_shared` immediately before the release is what makes
        "this thread is done reading the stage" true at the point it is signalled. MEASURED on the
        two-A ``x_gate`` path at ``M=32768``, 40 trials a shape: N=288 40/40, N=320 29/40,
        N=448 40/40 corrupt unfenced, **0/40 at all three fenced**.

        The race is each warp against its OWN outstanding loads, not warp against warp -- the
        mainloop pipeline takes one arrival per warp (``consumer_arrive_cnt = tiled_mma.size //
        WARP_SIZE``), so a warp releases only on its own behalf. That is why a fence suffices and
        `cute.arch.barrier` would be pure cost. Warp 0 is where it fires, and the asymmetry is the
        epilogue's: warp 0 is the store pipeline's TMA warp, so it enters the next mainloop last,
        arrives last, and the producer -- already waiting on the other seven -- refills instantly.

        **The fence is NOT placed here, and that is a blocked workaround rather than a judgement.**
        Putting it at the end of this function, or at either fold call site, SEGFAULTS THE COMPILER
        on the non-``x_gate`` paths -- a host-side fault inside the front door before any cubin
        exists, reproducible at ``alg_fold``/bf16/``chunk_g=1``/no-bias/no-mask, M=32768, N=K=128.
        So the fence lives at the ``x_gate`` call site alone, which is the only path where the
        defect is OBSERVED (the fold paths measure 0/20 unfenced at N=320 and N=448, clean by
        pipeline slack -- they stage two operands per stage where ``x_gate`` stages four on a
        halved ``ab_stage``). The fold paths therefore carry the same hazard, unfenced, latent.
        Fixing that needs the compiler fault resolved first; see the ledger.

        ``Sum x^2`` is accumulated directly rather than as a centred sum, because the mean is not
        known until every tile has been seen. The cancellation that implies is bounded here: the
        activation is a 16-bit value, so ``x^2`` is exact in fp32 and only the summation rounds.

    Args:
        thr_copy: This thread's slice of the statistics tiled copy. Must be the same partition
            :func:`make_row_stat_partials` sized the fragments from.
        sA_slice: One staged ``(tile_M, tile_K)`` A tile, READ ONLY. Passing the tile of a stage
            the pipeline has already released reads recycled data -- silently wrong statistics.
            Releasing it AFTER this returns is not sufficient either: see the fence note above.
        row_sum: The fp32 ``Sum x`` partials, updated in place AND returned.
        row_sqsum: The fp32 ``Sum x^2`` partials, likewise.

    Returns:
        ``(row_sum, row_sqsum)`` -- the same fragments, returned so the caller's k-loop can rebind
        them (the DSL's loop-carried values are values, not references).
    """
    tAsA = thr_copy.partition_S(sA_slice)  # (CPY, CPY_M, CPY_N)
    tArA = cute.make_rmem_tensor_like(tAsA)
    cute.autovec_copy(tAsA, tArA)
    x = tArA.load().to(Float32)
    tile_sum = x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1))
    tile_sqsum = (x * x).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1))
    for m in cutlass.range_constexpr(cute.size(row_sum)):
        row_sum[m] = row_sum[m] + tile_sum[m]
        row_sqsum[m] = row_sqsum[m] + tile_sqsum[m]
    return row_sum, row_sqsum


@cute.jit
def finalize_row_stats(
    stats_smem: cute.Tensor,
    tidx: Int32,
    row_sum,
    row_sqsum,
    n_elems,
    eps,
    threads_per_row: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    barrier,
    publish_fold_scale: cutlass.Constexpr[bool] = False,
    leading_barrier: cutlass.Constexpr[bool] = False,
) -> None:
    """Combine the per-lane partials into the row statistics and publish them to the shared scratch.

    Purpose
        The point where per-thread partials become a per-ROW fact the whole CTA can read. Splitting
        it from :func:`accumulate_row_stats` is what keeps the butterfly out of the k-loop.

    Semantics
        Butterflies the ``threads_per_row`` cooperating lanes together with ``shfl.bfly``, then

            ``mu = Sum / N``,  ``var = Sum2 / N - mu*mu``,  ``rstd = rsqrt(var + eps)``

        with the reciprocal ``1/N`` hoisted (so each is a multiply, not a divide) and ``rsqrt`` in
        fast-math form -- both as the upstream kernel has them, and both numerically observable.
        Only the lane owning a row writes it; the others computed the same value redundantly, which
        is cheaper than a predicated broadcast.

        **WHICH PAIR is published is a compile-time choice, because the two consumers want
        different pairs and one of them wants a product.** A normalize needs ``(mu, rstd)`` to
        rescale ``x``. The algebraic fold never rescales ``x``; its epilogue repairs the
        accumulator with ``r*acc - s*c + d`` where ``s = rstd*mu``, so publishing ``(rstd,
        rstd*mu)`` hands it that product already formed. Forming it HERE costs one multiply per
        ROW; forming it in the epilogue costs one per output ELEMENT -- measured at 512 extra warp
        instructions per 128x128 tile, ~9% of the kernel at K=128. The values are identical either
        way: both are the fp32 product of the same two fp32 scalars, so the switch is
        bit-preserving.

        **The leading barrier is `leading_barrier`, and whether it is emitted is a PER-CALLER fact
        about what the upstream kernel does -- not a judgement about whether it is needed.** A
        persistent CTA processes many output tiles through this ONE scratch buffer, so tile *n+1*'s
        writes must not overtake tile *n*'s reads. Two things can order them:

        * the EPILOGUE's own rendezvous -- cooperatively the barrier passed here IS
          ``epilogue_barrier``, and the epilogue arrives on it twice per subtile, after that
          subtile's statistics have been read, so every thread has passed a common barrier after
          its last read before any thread reaches the next tile's mainloop; and
        * an explicit LEADING ``arrive_and_wait()``, which the upstream `staged` kernel emits
          anyway, gated on its own ``_stats_race_guard`` (True by default, on its default
          ``stats_mode="streaming"``). Its comment records the race as OBSERVED, not theorised:
          *"a CTA that processes >1 tile at multi-wave M could clobber tile N's stats before tile N
          consumed them (the non-deterministic race seen in the sibling stage-C)"*.

        **An earlier revision of this file removed the leading barrier everywhere and justified it
        partly with "the kernel this reproduces has no such barrier either". That statement was
        FALSE for the `prolog_ln` callers.** ``_stats_race_guard`` is declared in the upstream
        `staged` kernel and applied at both of its finalize entries; the `stagec` and
        `layernorm_gemm_stagec` kernels contain no such guard at all. So the claim held for the
        algebraic-fold callers and not for the physical-prologue ones, and the two were changed
        together.

        It is restored HERE, per caller, on that evidence: `leading_barrier=True` at the two
        `prolog_ln` sites whose upstream counterpart emits it, and left False at the `alg_fold` and
        `layernorm_gemm` sites whose counterparts do not. Adding it to those would be adding
        synchronization the upstream lacks, which is the opposite error and is separately barred.

        **It is not free**, and the cost is recorded rather than hidden: one extra CTA-wide barrier
        per work tile, a constant against a mainloop whose length scales with K, measured at 5.1%
        at D=128 and 1.2% at D=512 -- NCU localized it as waiting rather than working (instructions
        +0.18%, cycles +4.13%, barrier-stall ratio 0.38 vs 0.28). Parity with the upstream is the
        bar; a 4% win taken by dropping its synchronization is not ours to keep.

        The TRAILING fence and barrier are not optional: they are what make the writes visible to
        the normalize. Every thread of the cooperating group must reach this call.

    Args:
        stats_smem: The ``(tile_M, 2)`` fp32 scratch. Which value lands in which column depends on
            `publish_fold_scale`, and NOTHING checks that the reader agrees -- a mismatch scales
            every row by the wrong statistic and raises nothing.
        tidx: This thread's index within the cooperating group; it selects the row group and the
            owner lane, so a value that is not reduced modulo the group size under a split schedule
            makes two warpgroups claim the same rows.
        row_sum: This thread's ``Sum x`` partials.
        row_sqsum: This thread's ``Sum x^2`` partials.
        n_elems: The per-row element count -- the TRUE feature width, not the padded one. Using the
            padded extent would divide by too large an N and shrink every mean.
        eps: The variance floor, added before the reciprocal square root. A RUNTIME value: it is a
            caller-supplied scalar and baking it would key the compiled artifact on it.
        threads_per_row: Lanes cooperating on one row. Must divide 32 and match the tiled copy.
        num_threads: Threads in the cooperating group. Must match the tiled copy.
        barrier: The named barrier the group shares. Must be the one the CONSUMER of these
            statistics also rendezvouses on -- cooperatively that is ``epilogue_barrier`` -- because
            that shared identity is what orders the next tile's writes after this tile's reads. A
            private barrier here would compile and would reintroduce the write-after-read race this
            function relies on the epilogue to close.
        publish_fold_scale: Which pair to publish. False (default) writes ``(mu, rstd)``, what
            :func:`normalize_tile` reads. True writes ``(rstd, rstd*mu)``, what the algebraic
            fold's epilogue reads. **Compile-time**, so exactly one pair of stores is emitted --
            it is not a runtime branch and costs nothing in the unused arm.
        leading_barrier: Emit the persistent-tile stats-race guard -- an ``arrive_and_wait()`` on
            `barrier` BEFORE anything is written. **Compile-time.** Set True only where the upstream
            kernel this caller reproduces emits it (`staged`, i.e. the `prolog_ln` variants); False
            elsewhere, because emitting it where the upstream does not is adding synchronization
            rather than restoring it. It must be the SAME barrier the consumer rendezvouses on, or
            it orders nothing.

    Returns:
        None. Its effect is ``stats_smem``, visible to the whole group on return.
    """
    # LEADING guard: the PREVIOUS tile's normalize read this same per-CTA scratch, and a persistent
    # CTA is about to overwrite it. See `leading_barrier` above for why this is per-caller.
    if const_expr(leading_barrier):
        barrier.arrive_and_wait()
    inv_n = 1.0 / Float32(n_elems)
    num_thread_rows = const_expr(num_threads // threads_per_row)
    row_grp = tidx // threads_per_row
    is_owner = tidx % threads_per_row == 0
    for m in cutlass.range_constexpr(cute.size(row_sum)):
        tile_sum = cute.arch.warp_reduction(
            row_sum[m], operator.add, threads_in_group=threads_per_row
        )
        tile_sqsum = cute.arch.warp_reduction(
            row_sqsum[m], operator.add, threads_in_group=threads_per_row
        )
        mean = tile_sum * inv_n
        var = tile_sqsum * inv_n - mean * mean
        rstd = cute.math.rsqrt(var + eps, fastmath=True)
        if is_owner:
            row = row_grp + m * num_thread_rows
            if const_expr(publish_fold_scale):
                stats_smem[row, 0] = rstd
                stats_smem[row, 1] = rstd * mean
            else:
                stats_smem[row, 0] = mean
                stats_smem[row, 1] = rstd
    cute.arch.fence_view_async_shared()
    barrier.arrive_and_wait()


@cute.jit
def normalize_tile(
    sA_stage: cute.Tensor,
    stats_smem: cute.Tensor,
    s_weight: cute.Tensor,
    s_bias: Optional[cute.Tensor],
    thr_copy,
    g_ktile: Int32,
    tile_shape_mk: cutlass.Constexpr,
    a_dtype: cutlass.Constexpr,
) -> None:
    """Rewrite one staged A tile IN PLACE as ``(x - mu) * rstd * gain[k] + bias[k]``.

    Purpose
        The "physical" half of the physical fusion: after this, the tile in shared memory IS
        ``LayerNorm(A)``, so the WGMMA that follows is an ordinary GEMM and every downstream stage
        -- the gate, the mask, the store, and later the all-to-all scatter -- needs no knowledge
        that a LayerNorm happened.

    Semantics
        In place, through registers: load this thread's owned elements, compute in fp32, store back
        in the ACTIVATION's dtype. Two consequences the caller owns. First, the tile is destroyed --
        a second pass over the same stage would normalize already-normalized data, so the caller
        must fence and barrier between this and the MMA, and must not re-enter it for the same
        stage. Second, the narrowing back to 16 bits is where the fusion's precision differs from
        an unfused LayerNorm followed by a GEMM: identical arithmetic, one extra rounding.

        ``gain``/``bias`` are indexed by the GLOBAL column ``g_ktile * tile_K + local_k``, not by
        the local one. That is why ``g_ktile`` is an argument and not derived: a caller whose
        pipeline stage order does not match its global k order (a staggered or composite-K loader)
        must pass the GLOBAL tile it actually staged, and passing the loop index instead applies
        the wrong slice of the gain -- a wrong answer with no diagnostic.

    Args:
        sA_stage: The staged ``(tile_M, tile_K)`` A tile, OVERWRITTEN.
        stats_smem: The ``(tile_M, 2)`` scratch :func:`finalize_row_stats` published -- column 0
            mean, column 1 reciprocal standard deviation. Reading it before that call's barrier
            gives the previous tile's statistics.
        s_weight: The staged ``(gemm_k,)`` fp32 gain. Must cover ``(g_ktile + 1) * tile_K``
            columns; see :func:`stage_ln_affine` for why that is the PADDED extent.
        s_bias: The staged ``(gemm_k,)`` fp32 bias, or None. None prunes the add entirely rather
            than adding a zero, so a kernel built without a bias cannot be handed one.
        thr_copy: This thread's slice of the statistics tiled copy -- the SAME partition the
            reduction used, so a thread rescales the rows whose statistics it helped compute.
        g_ktile: The GLOBAL k-tile index this stage holds. A runtime ``Int32``; see Semantics.
        tile_shape_mk: ``(tile_M, tile_K)``, a compile-time pair. Used to build the coordinate
            tensor that recovers each owned element's ``(m, k)``.
        a_dtype: The activation's element type, for the narrowing store back into shared memory.

    Returns:
        None. Its effect is ``sA_stage``.
    """
    has_bias = const_expr(s_bias is not None)
    blk_m, blk_k = const_expr(tile_shape_mk)
    cA = cute.make_identity_tensor((blk_m, blk_k))
    tAsA = thr_copy.partition_S(sA_stage)  # (CPY, CPY_M, CPY_N)
    tAcA = thr_copy.partition_S(cA)  # the same partition, of (m, k) coordinates
    tArA = cute.make_rmem_tensor_like(tAsA)
    cute.autovec_copy(tAsA, tArA)
    for i in cutlass.range_constexpr(cute.size(tArA)):
        m = tAcA[i][0]
        k = g_ktile * blk_k + tAcA[i][1]
        mu = stats_smem[m, 0]
        rstd = stats_smem[m, 1]
        val = (tArA[i].to(Float32) - mu) * rstd * s_weight[k]
        if const_expr(has_bias):
            val = val + s_bias[k]
        tArA[i] = val.to(a_dtype)
    cute.autovec_copy(tArA, tAsA)
