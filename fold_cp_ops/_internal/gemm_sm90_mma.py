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

# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.
"""The SM90 GEMM's compute layer: the WGMMA atom, the k-loop, and the MMA warpgroups.

Consumes staged SMEM tiles and produces a register-resident fp32 accumulator. It never touches a
global tensor and never stores -- the epilogue layer owns everything downstream of the
accumulator.

**Numerically this layer is exact for integer operands**, and the correctness gates lean on that:
bf16/fp16 products are exact in fp32, so if every partial sum stays under ``2**24`` the
accumulator is bit-identical to an integer reference regardless of how the k-loop was ordered.
That is what lets one equality gate every tile shape, cluster and pingpong split at once (see
``fold_cp_ops.testing.numerics``).

The ping-pong barriers live here rather than in a scheduling module because what they order is
this layer's hazard: two warpgroups alternating mainloop and epilogue over the same SMEM.
"""

from typing import Tuple, Type, Callable, Optional, Union, Literal  # noqa: F401
from functools import partial
import math  # noqa: F401

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp, warpgroup  # noqa: F401
import cutlass.utils.hopper_helpers as sm90_utils  # noqa: F401
from cutlass import Int32, Float32, Float16, Boolean, const_expr  # noqa: F401
from cutlass.utils import LayoutEnum  # noqa: F401

from fold_cp_ops._internal.pipeline import make_pipeline_state
import fold_cp_ops._internal.copy_utils as copy_utils
import fold_cp_ops._internal.sm90_utils as fold_cp_ops_sm90_utils
from fold_cp_ops._internal.rounding import RoundingMode  # noqa: F401


import enum


class NamedBarrierGemm(enum.IntEnum):
    """Named-barrier ids used by this kernel, offset so id 0 stays free.

    Hardware named barriers are a scarce, globally numbered resource within a CTA, and barrier 0 is
    reserved for ``__syncthreads``. Enumerating them here is what keeps two warp roles from picking
    the same number: ``enum.auto()`` starts at 1 for exactly that reason.

    ``EpilogueLoad`` exists to sequence rather than to synchronize: it lets the mainloop's load
    warps tell the epilogue load warp when it may start on C, which stops a C fetch from competing
    with A/B for TMA bandwidth before the mainloop needs it.
    """

    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    # For mainloop load warps to signal that the epilogue load warp can start.
    # This is to avoid loading C too early, interfering with loading A and B.
    EpilogueLoad = enum.auto()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()
    TmemPtr = enum.auto()


class MmaFragments:
    """The register-resident state one MMA warpgroup builds ONCE, before its work-tile loop.

    Returned by :meth:`GemmSm90MmaMixin.mma_setup_fragments` and consumed by
    :meth:`mma_consume_work_tile` and the epilogue. It exists so a derivation can enrich that state
    without re-implementing the role: the fused-LayerNorm dual-gated kernel needs a LIST of
    accumulators (one per output-N sub-tile, sharing one ``tCrA``) plus a statistics tiled-copy and
    the staged gain/bias, and on ``main`` the only way to get them was to fork ``kernel()``.

    A plain Python object, built and read at trace time, so it costs nothing at runtime. It is NOT
    passed across a ``@cute.jit`` boundary -- the DSL flattens arguments into MLIR values and cannot
    accept an object (docs/refactor_and_fix.md 3.15 rule 2) -- which is why every method taking one is
    a plain ``def``.

    ``__slots__`` is the check that a typo fails at trace time instead of silently reading ``None``
    and producing a kernel that multiplies nothing.

    Attributes:
        thr_mma: This warpgroup's slice of the tiled MMA.
        acc: The fp32 accumulator fragment. A subclass with several sub-tiles puts its FIRST here,
            so anything reading ``frags.acc`` keeps working.
        acc_slow: The second accumulator used by the fp8 slow-accumulate path, else None.
        tCrA: The A register fragment.
        tCrB: The B register fragment.
        mma_fn: The partially-applied WGMMA closure the k-loop calls per k-tile.
        extra: Free slot for a subclass's own state (a stats tiled-copy, staged gain/bias, an
            accumulator list). None on the base, and nothing in the base ever reads it.
    """

    __slots__ = ("thr_mma", "acc", "acc_slow", "tCrA", "tCrB", "mma_fn", "extra")

    def __init__(self, **fields):
        """Bind every slot by keyword.

        Args:
            **fields: One entry per name in ``__slots__`` except ``extra``, which defaults to None.
                Keyword-only because ``acc`` and ``acc_slow`` are the same shape and type, so a
                positional form would let them be swapped with no diagnostic.

        Raises:
            TypeError: If a required slot is missing.
            AttributeError: If a name is not a declared slot.
        """
        fields.setdefault("extra", None)
        missing = set(self.__slots__) - set(fields)
        if missing:
            raise TypeError(f"MmaFragments missing field(s): {sorted(missing)}")
        for k, v in fields.items():
            setattr(self, k, v)


class MmaContext:
    """The launch-scoped values a derivation's MMA state needs, which the role alone does not carry.

    Purpose
        :meth:`GemmSm90MmaMixin.mma_setup_fragments` takes the tiled MMA and the two staged SMEM
        tiles -- everything a plain GEMM's fragments are built from. A FUSED mainloop needs more:
        the fused-LayerNorm variants both accumulate per-row statistics into an SMEM scratch that
        the *epilogue* allocated (through its declared ops) and read ``eps`` off the *epilogue*
        arguments. Neither is reachable from the role's other parameters, and stashing them on
        ``self`` is exactly the mistake :mod:`compile_time.template_params` exists to prevent -- they
        carry MLIR values.

    Semantics
        A trace-time bundle, like :class:`MmaFragments` and ``_KernelContext``: constructing it
        emits no instructions and costs no registers. It is built once per kernel, in
        :meth:`mma_warpgroup_role`, and handed to :meth:`mma_setup_fragments`, whose override may
        keep whatever it needs in ``MmaFragments.extra``. It is NOT passed across a ``@cute.jit``
        boundary -- the DSL flattens arguments into MLIR values and cannot accept an object -- which
        is why :meth:`mma_setup_fragments` is a plain ``def``.

        ``__slots__`` makes a typo fail at trace time rather than silently reading ``None``.

    Attributes:
        storage: The allocated ``SharedStorage`` instance, for a derivation whose extra SMEM came
            from ``_extra_smem_struct`` rather than from an epilogue op.
        epi_smem_tensors: The epilogue ops' SMEM tensors, in declaration order. Index it with
            ``self._epi_smem_map[name]``; reading it positionally breaks the moment an op is added.
        epilogue_params: The lowered epilogue params. The only lawful home for a scalar the
            mainloop needs but the caller supplies per launch (``eps``).
        tidx: This thread's index within the CTA, already reduced modulo the warpgroup size under
            ping-pong. A per-row reduction partitions by it, so passing the un-reduced value would
            give two warpgroups the same rows.
    """

    __slots__ = ("storage", "epi_smem_tensors", "epilogue_params", "tidx")

    def __init__(self, **fields):
        """Bind every slot by keyword.

        Args:
            **fields: One entry per name in ``__slots__``. All required and keyword-only.

        Raises:
            TypeError: If a slot is missing.
            AttributeError: If a name is not a declared slot.
        """
        missing = set(self.__slots__) - set(fields)
        if missing:
            raise TypeError(f"MmaContext missing field(s): {sorted(missing)}")
        for k, v in fields.items():
            setattr(self, k, v)


class GemmSm90MmaMixin:
    def _make_tiled_mma(self, b_dtype, a_layout, b_layout) -> cute.TiledMma:
        """Build the WGMMA atom this configuration issues, for these operands. Override for other MMAs.

        Purpose
            The atom is the one object that decides both what the MMA consumes (its K extent, which
            becomes ``cta_tile_k``) and how accumulators land in registers (its N permutation). It
            carries MLIR values, so it is RETURNED rather than stashed on ``self`` -- a stashed MLIR
            value does not survive the ``@cute.jit`` -> ``@cute.kernel`` boundary, and the caller
            already passes this one to ``kernel()`` as an explicit argument.

        Semantics
            The operand facts arrive as arguments rather than off ``self`` because this runs BEFORE
            the call parameters are bound: ``cta_tile_k`` is read back off the atom built here, so
            the atom cannot wait for the pack that will contain it.

            When N is split across two warpgroups the N mode is permuted, so that in the epilogue
            WG0 and WG1 write to one contiguous epilogue SMEM tile (e.g. 64x32) instead of two
            distant 64x16 halves.

        Args:
            b_dtype: B's element type. Must match ``self.a_dtype``'s width; the caller checks that.
            a_layout: A's major mode, as a ``LayoutEnum``.
            b_layout: B's major mode.

        Returns:
            The ``cute.TiledMma``. Its ``shape_mnk`` K extent times ``_MMA_INST_TILE_K`` is the
            mainloop K tile.

        Note:
            Emits MLIR, so it needs a Context/Module/InsertionPoint. ``__call__`` is ``@cute.jit``
            and supplies all three; a direct caller in a test must supply them itself.
        """
        tile_N = self.tile_shape_mn[1]
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            b_dtype,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, tile_N // self.atom_layout_mnk[1]),
        )
        if const_expr(self.atom_layout_mnk[1] > 1):
            # If N dimension is split among 2 WGs, we need to permute the N dimension so
            # that in the epilogue, WG0 and WG1 can write to epi smem of size e.g. (64, 32)
            # containing accumulators that are next to each other in the N dimension.
            # Without permutation WG0 would write to epi smem of size (64, 16) and
            # WG1 would write to a separate epi smem of size (64, 16) that's far away.
            atom_n = self.atom_layout_mnk[1]
            permutation_n = cute.make_ordered_layout(
                (8, tile_N // atom_n // 8, atom_n), order=(0, 2, 1)
            )
            tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(tiled_mma.op),
                self.atom_layout_mnk,
                permutation_mnk=(None, permutation_n, None),
            )
        return tiled_mma

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        acc: cute.Tensor,
        acc_slow: Optional[cute.Tensor],
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        on_ktile: Optional[Callable] = None,
        k_tile_cnt_const: cutlass.Constexpr = None,
        on_drain: Optional[Callable] = None,
    ) -> cutlass.pipeline.PipelineState:
        # Prologue MMAs
        """Consumer loop: wait for each staged k-tile and issue its WGMMA into the accumulator.

        The first k-tile overwrites the accumulator and the rest accumulate, which is why the
        underlying issue takes a **runtime** ``zero_init`` -- in a persistent kernel "which tile is
        first" is not known at trace time.

        Args:
            ab_pipeline: The A/B pipeline.
            ab_read_state: The consumer state, advanced per k-tile and returned.
            mma_fn: Closure issuing one k-tile's MMA chain.
            acc: The accumulator, updated in place.
            acc_slow: The fp32 shadow accumulator for the fp8 slow-accumulation path, or None. When
                present, ``acc`` is promoted into it periodically to bound the error growth that
                8-bit accumulation would otherwise incur.
            k_tile_cnt: Number of k-tiles for this work tile.
            k_tile_cnt_const: The SAME count as a compile-time Python int, or None to keep the
                dynamic loop. **Not a scheduling preference.** A dynamic ``cutlass.range`` is a
                separate MLIR region, so register state a caller accumulates ACROSS k-tiles through
                `on_ktile` cannot stay straight-line SSA -- it is carried through the region and
                spills. The fused-LayerNorm statistics are exactly that state, and the upstream
                unrolls this loop for exactly this reason. Measured on the algebraic fold: the
                dynamic loop costs ~10% cooperatively and far more under ping-pong, where each
                thread carries twice the partials.
                **It is not free.** Unrolling emits the WGMMA chain once per k-tile, so the compile
                cost grows with K -- the known root-cause class in this package's compile-time rule.
                Pass it only where the runtime win is measured and the compile cost is accepted.
                None keeps the existing path, `const_expr`-pruned, so every caller that does not
                ask is byte- and PTX-identical.
            warp_group_idx: Which MMA warpgroup this is, for the ping-pong barriers.
            on_ktile: Optional ``on_ktile(stage_index)``, called once per k-tile with the pipeline
                stage that tile is staged in, AFTER its WGMMA is issued and BEFORE the group wait --
                so an extra read of the staged tile overlaps the flying WGMMA rather than serializing
                with the MMA warps' own operand fetch. **It must only READ**, and only from the stage
                it is given: the buffer is still owned by this iteration, but the MMA is in flight
                over it. The fused-LayerNorm statistics reduction is what this exists for.

                None (the default) is ``const_expr``-pruned, so a plain GEMM's mainloop is unchanged.
            on_drain: Optional zero-argument closure run AFTER the mainloop and BEFORE the drain's
                ``wait_group(0)`` -- i.e. while the last WGMMAs are still in flight.

                **This placement is the whole point, and it is worth ~4.7% on the fused fold.**
                ``wait_group(0)`` waits for every outstanding WGMMA, so whatever sits between the
                final MMA issue and that wait is free: it executes in the shadow. With nothing
                there, the wait is naked and the warp stalls the full WGMMA latency. Measured on
                the algebraic fold at D=128, NCU: the drain ``WARPGROUP.DEPBAR.LE gsb0, 0x0``
                carried 240 barrier-stall samples with the two waits 2 instructions apart, against
                0 for the kernel this reproduces, which has 49 instructions of epilogue address
                setup between them. The upstream states the same intent in a comment -- "finalize
                stats into the SMEM scratch while the last WGMMAs drain".

                The closure must be INDEPENDENT of the accumulator: it runs before the MMA is known
                to have retired, so anything reading ``acc`` here reads a partial result. The
                fused-LayerNorm statistics finalize qualifies -- it touches register partials and
                the SMEM stats scratch, neither of which the WGMMA writes.

                None (the default) is ``const_expr``-pruned, so every other caller is unchanged.

        Returns:
            The advanced consumer state.
        """
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        # With a static count the prologue bound must be static too, or `range_constexpr` below
        # gets a dynamic start and refuses.
        num_prologue_mma = (
            min(k_pipe_mmas, k_tile_cnt)
            if const_expr(k_tile_cnt_const is None)
            else min(k_pipe_mmas, k_tile_cnt_const)
        )
        if const_expr(self.pingpong):
            self.pingpong_barrier_sync(warp_group_idx, stage="mma")
        peek_ab_full_status = Boolean(True)
        zero_init = Boolean(True)
        # PROLOGUE, written TWICE for the same reason the mainloop below is, and for one more:
        # given a STATIC count, every loop bound AND every look-ahead guard must be compared
        # against `k_tile_cnt_const`, never against the runtime `k_tile_cnt`. A guard that mixes a
        # `range_constexpr` induction variable with the runtime count is still DYNAMIC -- it emits
        # a compare and a conditional `try_wait` region per unrolled iteration, which is precisely
        # the unrolling the static count was passed to avoid. Measured: ~9 extra branches and ~37
        # extra instructions per k-tile, plus a fixed per-work-tile cost from the dynamic prologue
        # and drain loops. The upstream compares against its static count throughout.
        if const_expr(k_tile_cnt_const is None):
            if 0 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
            for k_tile in cutlass.range(num_prologue_mma):
                # Wait for A/B buffer to be ready
                ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
                mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
                if const_expr(on_ktile is not None):
                    on_ktile(ab_read_state.index)
                zero_init = Boolean(False)
                ab_read_state.advance()
                peek_ab_full_status = Boolean(True)
                if k_tile + 1 < k_tile_cnt:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        else:
            if const_expr(k_tile_cnt_const > 0):
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
            for k_tile in cutlass.range_constexpr(num_prologue_mma):
                # Wait for A/B buffer to be ready
                ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
                mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
                if const_expr(on_ktile is not None):
                    on_ktile(ab_read_state.index)
                zero_init = Boolean(False)
                ab_read_state.advance()
                peek_ab_full_status = Boolean(True)
                if const_expr(k_tile + 1 < k_tile_cnt_const):
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        # If k_tile_cnt == 0, this is not correct. But we will set acc to 0 in the mainloop
        # in that case.
        if const_expr(self.fp8_slow_accum):
            warpgroup.wait_group(0)
            acc_slow.store(acc.load())

        # MAINLOOP, written TWICE on purpose. The two bodies are identical and must stay so; the
        # only difference is the loop construct, and the duplication is not avoidable by factoring
        # the body into a local function -- the DSL re-derives this method from source, so a nested
        # closure loses its free variables and fails with "cannot access local variable
        # 'ab_pipeline'". Same class as the zero-argument `super()` trap.
        if const_expr(k_tile_cnt_const is None):
            for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
                ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
                if const_expr(self.fp8_slow_accum):
                    zero_init = Boolean(True)
                mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
                if const_expr(on_ktile is not None):
                    on_ktile(ab_read_state.index)
                zero_init = Boolean(False)
                # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
                if const_expr(not self.fp8_slow_accum):
                    warpgroup.wait_group(k_pipe_mmas)
                else:
                    warpgroup.wait_group(0)
                    acc_slow.store(acc_slow.load() + acc.load())
                ab_pipeline.consumer_release(ab_release_state)
                ab_read_state.advance()
                ab_release_state.advance()
                peek_ab_full_status = Boolean(True)
                if k_tile + 1 < k_tile_cnt:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        else:
            for k_tile in cutlass.range_constexpr(num_prologue_mma, k_tile_cnt_const):
                ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
                if const_expr(self.fp8_slow_accum):
                    zero_init = Boolean(True)
                mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
                if const_expr(on_ktile is not None):
                    on_ktile(ab_read_state.index)
                zero_init = Boolean(False)
                # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
                if const_expr(not self.fp8_slow_accum):
                    warpgroup.wait_group(k_pipe_mmas)
                else:
                    warpgroup.wait_group(0)
                    acc_slow.store(acc_slow.load() + acc.load())
                ab_pipeline.consumer_release(ab_release_state)
                ab_read_state.advance()
                ab_release_state.advance()
                peek_ab_full_status = Boolean(True)
                if const_expr(k_tile + 1 < k_tile_cnt_const):
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        if const_expr(self.pingpong):
            # Cue for next WG's MMA to start
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        # Accumulator-independent tail work, run while the last WGMMAs are still in flight so the
        # drain below finds them retired. See `on_drain` in the docstring for the measurement.
        if const_expr(on_drain is not None):
            on_drain()
        if const_expr(not self.fp8_slow_accum):
            # fp8_slow_accum would already called wait_group(0) inside the loop
            warpgroup.wait_group(0)
        # DRAIN, twice for the same reason: with a static count `num_prologue_mma` is a Python int
        # and the trip count folds; with a dynamic one it is a runtime value and cannot.
        if const_expr(k_tile_cnt_const is None):
            for k_tile in cutlass.range(num_prologue_mma, unroll=1):
                ab_pipeline.consumer_release(ab_release_state)
                ab_release_state.advance()
        else:
            for k_tile in cutlass.range_constexpr(num_prologue_mma):
                ab_pipeline.consumer_release(ab_release_state)
                ab_release_state.advance()
        if const_expr(self.fp8_slow_accum):
            acc.store(acc_slow.load())
        return ab_read_state

    def mma_initial_carry(self):
        """Initial value of the state :meth:`mma_consume_work_tile` carries between work tiles.

        **A separate hook because the DSL requires one.** The carry is a loop-carried variable of a
        *dynamic* ``while``, and the DSL refuses a type change across one: initialising to ``None``
        and returning an ``Int32`` from the first iteration fails with "`mma_carry` is None prior to
        this `while`, and update to Int32 inside of this `while` is not supported". So a derivation
        with a real carry must seed it with a value of the SAME type it will return -- which is
        exactly what the fused-LayerNorm kernel does on ``main``, priming its reduction phase at 0
        and its empty-barrier phase at 1 before the loop starts.

        Returns:
            ``None`` on the base, which carries nothing. A derivation returns whatever
            :meth:`mma_consume_work_tile` will return -- typically a tuple of barrier phases.
            Whatever it is, it must be type-stable across iterations.
        """
        return None

    def mma_setup_fragments(
        self, tiled_mma, sA, sB, warp_group_thread_layout, warp_group_idx, mma_ctx=None
    ):
        """Build the accumulator and operand fragments this warpgroup will reuse for every tile.

        Runs ONCE, outside the work-tile loop, because the fragments are registers: re-deriving them
        per tile would emit the same layout algebra again for nothing, and the accumulator must
        persist so the k-loop can accumulate into it.

        **The seam.** A derivation that needs different register state overrides this and returns an
        enriched :class:`MmaFragments` -- the fused-LayerNorm dual-gated kernel returns one
        accumulator per output-N sub-tile in ``extra`` (they share ``tCrA``, since it is the same A
        stream) plus its statistics tiled-copy and staged gain/bias. Overriding here replaces ~15
        lines instead of the whole role.

        Args:
            tiled_mma: The MMA atom. Sliced by warpgroup, not by thread -- WGMMA's per-thread
                register mapping is internal to the atom.
            sA: The staged A tile in SMEM, which sets ``tCrA``'s layout.
            sB: The staged B tile, likewise for ``tCrB``.
            warp_group_thread_layout: Maps a warpgroup index to its first thread. Built by the
                caller because it depends on ``pingpong``.
            warp_group_idx: Which MMA warpgroup this is. Ignored under pingpong, where both
                warpgroups slice at 0 and alternate over the SAME accumulator space instead of
                splitting M.
            mma_ctx: An :class:`MmaContext`, or None. The base ignores it entirely -- it exists so a
                FUSED mainloop can reach the epilogue's SMEM scratch, the epilogue params and this
                thread's index, none of which the other arguments carry.

                **An override MUST accept it**, even to ignore it: the role passes it POSITIONALLY,
                so a five-parameter override raises ``TypeError`` at trace time. The default exists
                for DIRECT callers (tests that drive the seam without a role), not for signature
                compatibility -- there is no way to add a parameter a positional call site can omit.

        Returns:
            An :class:`MmaFragments`. ``acc_slow`` is non-None only on the fp8 slow-accumulate path,
            where it holds the second accumulator the k-loop alternates into.

        Note:
            **This is a plain ``def`` and an override must be too.** ``@cute.jit`` flattens every
            argument into MLIR values and cannot accept or return an :class:`MmaFragments`, so a
            decorated override fails with ``DSLTreeFlattenError``. The consequence for an override
            is that it does NOT get the DSL preprocessor: ``const_expr`` still works (it is an
            ordinary function that returns its argument), but ``cutlass.range_constexpr`` raises
            "should be preprocessed by preprocessor" -- use a plain ``range`` over a
            compile-time-constant trip count, which unrolls identically at trace time.
        """
        thr_mma = tiled_mma.get_slice(
            warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
        )
        acc, tCrA, tCrB = fold_cp_ops_sm90_utils.partition_fragment_ABC(
            thr_mma, self.cta_tile_shape_mnk, sA, sB
        )
        acc_slow = None
        if const_expr(self.fp8_slow_accum):
            acc_slow = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
        return MmaFragments(
            thr_mma=thr_mma,
            acc=acc,
            acc_slow=acc_slow,
            tCrA=tCrA,
            tCrB=tCrB,
            mma_fn=partial(fold_cp_ops_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB),
        )

    def mma_consume_work_tile(
        self, frags, ab_pipeline, ab_read_state, len_k, warp_group_idx, mma_carry=None
    ):
        """Drain one work tile's k-tiles into the accumulator.

        A one-call wrapper around :meth:`mma`, and the wrapper is the point: it is the seam a
        derivation replaces to do something *else* per work tile while keeping the role's loop,
        scheduling and epilogue. The fused-LayerNorm kernel substitutes its two-pass
        statistics-plus-normalize variants here.

        **``mma_carry`` is what makes that possible.** It is loop-carried across work tiles by the
        caller: a subclass returns state it needs on the NEXT tile (the dual-gated cluster path
        carries its reduction and empty-barrier phases this way) and gets it back on the following
        call. The base neither produces nor reads it, so it stays the Python constant ``None`` and
        is erased at trace time.

        Args:
            frags: The :class:`MmaFragments` from :meth:`mma_setup_fragments`.
            ab_pipeline: The mainloop staging pipeline.
            ab_read_state: The consumer pipeline state, threaded across work tiles. Must be the
                value the previous call returned -- reusing a stale one reads an already-released
                stage.
            len_k: The contraction extent, converted to a k-tile count through ``_k_tile_cnt``.
                **The producer must use the same hook**, or it stages a different number of tiles
                than this drains, which is a deadlock rather than a wrong answer.
            warp_group_idx: Which MMA warpgroup, for the pingpong alternation.
            mma_carry: Whatever the previous call returned as carry, or None on the first tile.

        Returns:
            ``(ab_read_state, mma_carry)`` -- the advanced pipeline state and the carry for the next
            work tile. The base always returns ``None`` for the carry.
        """
        k_tile_cnt = self._k_tile_cnt(len_k)
        ab_read_state = self.mma(
            ab_pipeline,
            ab_read_state,
            frags.mma_fn,
            frags.acc,
            frags.acc_slow,
            k_tile_cnt,
            warp_group_idx,
        )
        return ab_read_state, mma_carry

    @cute.jit
    def run_epilogue(
        self,
        tiled_mma,
        acc,
        sD,
        sC,
        has_C: cutlass.Constexpr[bool],
        copy_D,
        copy_C,
        epilogue_params,
        epi_smem_tensors,
        epi_pipeline,
        epi_store_pipeline,
        epi_read_state,
        epi_producer_state,
        tile_coord_mnkl,
        tile_scheduler,
        tidx,
        is_tma_warp,
        epi_gate3: cutlass.Constexpr = False,
    ):
        """Partition one work tile's accumulator for the store and run the epilogue over it.

        Purpose
            An extract-method seam, and the DEFAULT is byte-identical to the three statements it
            replaced. It exists so a derivation can decide, PER WORK TILE, *which* epilogue a tile
            gets -- the fused TriMul output gate routes a tile at or beyond the dual region to a
            different store, with a re-based coordinate -- without re-implementing
            :meth:`mma_warpgroup_role`. Overriding the epilogue HOOKS is not enough for that: they
            are called from inside :meth:`epilogue`, after the destination has already been chosen.

        Semantics
            Three steps, in the only order that works: partition the accumulator for the store
            (:meth:`epilogue_partition`), let the accumulator-level hook see it
            (:meth:`epi_visit_acc`), then run the subtile loop. A derivation that dispatches between
            two epilogues calls this method for each arm rather than inlining one, because the DSL
            forbids a variable-capturing closure inside dynamic control flow -- a method taking
            explicit arguments is the form that survives.

        Args:
            tiled_mma: The MMA atom, for the store partitioning.
            acc: This work tile's accumulator.
            sD: D's epilogue SMEM tile, or None.
            sC: C's SMEM tile, or None.
            has_C: Whether a C addend is present.
            copy_D: The D store closure, or None.
            copy_C: The C load closure, or None.
            epilogue_params: The lowered epilogue params.
            epi_smem_tensors: The epilogue ops' SMEM tensors.
            epi_pipeline: The C-load pipeline, or None.
            epi_store_pipeline: The D/post-activation store pipeline.
            epi_read_state: The C-load consumer state.
            epi_producer_state: The C-load producer state.
            tile_coord_mnkl: This work tile's coordinate. A derivation dispatching on the N block
                passes a RE-BASED coordinate to the arm it selects, so the store addresses its own
                tensor's origin rather than the combined operand's.
            tile_scheduler: The scheduler, for the store pipeline's tail accounting.
            tidx: This thread's index.
            is_tma_warp: Whether this warp issues the TMA store.
            epi_gate3: Which epilogue this invocation is, forwarded to :meth:`epilogue` and from
                there to every hook that branches on it. False on the base path, and
                ``const_expr``, so nothing about the plain GEMM changes.

        Returns:
            ``(epi_read_state, epi_producer_state)`` -- the advanced C-load pipeline states.
        """
        # Partition for the store -- the seam a derivation calls ONCE PER ACCUMULATOR when
        # it produces several per work tile (the dual-gated per-output-N-sub-tile pass).
        epi = self.epilogue_partition(tiled_mma, acc, sD, sC, tidx, has_C)
        self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)
        return self.epilogue(
            epilogue_params,
            epi_smem_tensors,
            epi_pipeline,
            epi_store_pipeline,
            epi_read_state,
            epi_producer_state,
            self.epi_tile,
            epi.load_acc_subtile,
            epi.tRS_rD,
            epi.tRS_rC,
            None,  # tiled_copy_t2r, for Sm100 only
            epi.tiled_copy_r2s,
            epi.tRS_sD,
            epi.tiled_copy_s2r,
            epi.tSR_rC,
            epi.tSR_sC,
            copy_D,
            copy_C,
            tile_coord_mnkl,
            self.epilogue_barrier,
            tile_scheduler,
            tidx,
            is_tma_warp,
            epi_gate3=epi_gate3,
        )

    @cute.jit
    def mma_warpgroup_role(
        self,
        warp_idx,
        tiled_mma,
        mA_mkl,
        mD_mnl,
        mC_mnl,
        tma_atom_d,
        tma_atom_c,
        epilogue_params,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        ab_pipeline,
        epi_pipeline,
        epi_smem_tensors,
        has_C: cutlass.Constexpr[bool],
        has_D: cutlass.Constexpr[bool],
        len_k,
        sA,
        sB,
        sC,
        sD,
        storage,
    ):
        """The MMA warpgroups: consume the staged tiles, accumulate, and run the epilogue.

        Runs only on ``warp_idx < ab_load_warp_id``. Raises its register budget first (the
        accumulator fragment is the kernel's register pressure), partitions the MMA fragments once
        outside the tile loop, then per tile: :meth:`mma` drains ``k_tile_cnt`` staged k-tiles into
        the accumulator, the epilogue partitioning is rebuilt, and :meth:`epilogue` stores.

        **The register reallocation is paired with the producer's and must stay paired.** Both are
        gated on the same ``_skip_warpgroup_reg_realloc`` flag so they are emitted together or not
        at all -- a lone ``setmaxnreg.inc`` with no matching ``.dec`` is ISA-undefined and, in the
        one measured case, blocks forever waiting for registers no warp released.

        Under pingpong the two warpgroups alternate mainloop and epilogue across successive tiles,
        which is why the second one advances its pipeline states past the first's before entering
        the loop and skips a tile on each iteration.

        Args:
            warp_idx: Warp-uniform index; the role guard and the pingpong warpgroup split read it.
            ctx: The :class:`_KernelContext` from :meth:`kernel_prologue`.
            tiled_mma: The MMA atom, for fragment partitioning.
            mA_mkl: A operand, read only for the static k-tile count.
            mD_mnl: Output tensor, or None.
            mC_mnl: Addend, or None.
            tma_atom_d: TMA atom for the D store, or None.
            tma_atom_c: TMA atom for the C load, or None.
            epilogue_params: Underlying epilogue params.
            tile_sched_params: Scheduler params, forwarded to the tail-drain hook.

        Returns:
            None. Its effect is the store into ``mD_mnl``.
        """
        if warp_idx < self.ab_load_warp_id:
            # Paired with the load/consumer setmaxregister_decrease above — both gated by the
            # same const_expr so they are emitted together or skipped together (a lone .inc/.dec
            # would be ISA-UB). Default OFF -> byte-identical (the realloc is always emitted).
            if const_expr(not getattr(self, "_skip_warpgroup_reg_realloc", False)):
                cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            # Partition global tensor for TiledMMA_A/B/C
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups if const_expr(not self.pingpong) else 1,
                stride=self.num_threads_per_warp_group,
            )
            # Make fragments -- the seam a derivation overrides to add register state (a second
            # accumulator, fused-LayerNorm statistics) without re-implementing this role.
            frags = self.mma_setup_fragments(
                tiled_mma,
                sA,
                sB,
                warp_group_thread_layout,
                warp_group_idx,
                MmaContext(
                    storage=storage,
                    epi_smem_tensors=epi_smem_tensors,
                    epilogue_params=epilogue_params,
                    tidx=tidx,
                ),
            )
            acc = frags.acc
            mma_carry = self.mma_initial_carry()

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            # CONSUMER K-loop trip = _k_tile_cnt (mirror the producer :957) so the composite-K subclass's
            # cp*nt_within count reaches the MMA consumer + the pingpong advance_iters too -- else the
            # producer loads cp*nt_within tiles while the consumer drains only nt_within => pipeline
            # DEADLOCK. shape[1] == len_k. DEFAULT ceil_div(shape[1], BLK_K) -> BYTE-IDENTICAL.
            k_tile_cnt_static = self._k_tile_cnt(mA_mkl.shape[1])
            c_tile_cnt = cute.size(cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile))

            ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_c_stage
            )
            epi_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.epi_c_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            if const_expr(self.pingpong):
                if warp_idx >= 4:
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    ab_read_state.advance_iters(k_tile_cnt_static)
                    # TODO: do we need to check if work_tile is valid?
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                ab_read_state, mma_carry = self.mma_consume_work_tile(
                    frags, ab_pipeline, ab_read_state, len_k, warp_group_idx, mma_carry
                )

                # EPILOGUE
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")

                copy_D = None
                if const_expr(has_D):
                    copy_D = self.build_D_copy_fn(
                        tma_atom_d,
                        mD_mnl,
                        batch_idx,
                        sD,
                        tile_coord_mnkl,
                        epilogue_params,
                        storage,
                    )
                copy_C = None
                if const_expr(has_C):
                    copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_c,
                        self.select_batch(mC_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sC,
                        tile_coord_mnkl,
                    )
                    copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)

                epi_read_state, epi_producer_state = self.run_epilogue(
                    tiled_mma,
                    acc,
                    sD,
                    sC,
                    has_C,
                    copy_D,
                    copy_C,
                    epilogue_params,
                    epi_smem_tensors,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    epi_producer_state,
                    tile_coord_mnkl,
                    tile_scheduler,
                    tidx,
                    is_tma_warp,
                )

                if const_expr(self.pingpong):
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signaling
                    # the next WG's epilogue.
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")

                if const_expr(not self.pingpong):
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                else:  # Skip a tile for pingpong
                    # Update starting load/store pipeline states for the next tile
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    # Update starting mainloop pipeline state for the next tile
                    ab_read_state.advance_iters(k_tile_cnt_static)
                    tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                    work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Wait for D store complete
            if const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()
                    # WIDE-PUT (front A2A ib_wide): the producer (warp 0) accumulates W subtiles per
                    # wide ring slot and flushes on break/W-cap; the CTA's TRAILING (partial) batch
                    # has no subsequent subtile to flush it, so flush it HERE (after the MMA tile
                    # loop) -> the consumer's final full[slot] lands and count-parity closes. const_
                    # expr-gated on the wide A2A path -> DEFAULT-OFF BYTE-IDENTICAL (the branch is
                    # elided at trace otherwise).
                    #
                    # Placement differs from the kernel this ports from, and the difference is
                    # forced: THERE the staged dual owns its own `kernel`, so the seam sits in that
                    # one method; HERE every fusion shares this mainloop, so the call site is
                    # visible to every kernel. It stays inert for all of them because the FIRST
                    # conjunct is absent everywhere but the front A2A subclass, and Python's `and`
                    # short-circuits -- which is also what keeps `_decoupled_active` from having to
                    # exist on a parent that has no use for it. `_a2a_ib_wide` being written by the
                    # front kernel alone is what makes that safe, so it is pinned by
                    # tests/_internal/test_gemm_sm90_mma.py rather than left to a grep.
                    #
                    # Under pingpong this is unreachable, and that matches: the staged dual asserts
                    # `not self.pingpong` (cooperative-only), so no pingpong execution of it exists
                    # to carry the seam in the first place.
                    if const_expr(
                        getattr(self, "_a2a_enabled", False)
                        and getattr(self, "_a2a_ib_wide", False)
                        and self._decoupled_active()
                    ):
                        self._a2a_wide_producer_finalize(storage)

            # Two-phase all-hands CLAIM-DRAIN TAIL (subclass hook; see GemmSm90A2A). The MMA warps,
            # having finished their persistent tile loop (so every full[] they produce has arrived),
            # fall into the per-CTA slot-claim loop to drain the remaining write-once ring slots ->
            # reclaims the idle MMA warps for the isolated drain tail. getattr(...,False) is a
            # compile-time const -> on the plain parent (no _a2a_drain_tail) this branch is elided at
            # trace and the PTX is byte-identical.
            if const_expr(getattr(self, "_a2a_drain_tail", False)):
                self.tail_drain_role(warp_idx, storage, epilogue_params, tile_sched_params)

    def pingpong_barrier_sync(self, warp_group_idx: Int32, stage: Literal["mma", "epi"]):
        """Block this warpgroup on its ping-pong barrier for the given stage.

        Ping-pong runs two MMA warpgroups out of phase: while one is in its mainloop the other is in
        its epilogue. Two barrier pairs (``MmaWG*`` and ``EpiWG*``) enforce that alternation, and
        the ordering is the schedule -- a missed sync lets both warpgroups into the same stage and
        they contend for the same SMEM.

        Args:
            warp_group_idx: Which MMA warpgroup this is (0 or 1). It selects the barrier, so a
                wrong value makes the two warpgroups wait on each other's barrier and hang.
            stage: ``"mma"`` or ``"epi"``.

        Returns:
            None. Every thread of the warpgroup must reach this call.
        """
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def pingpong_barrier_arrive(self, warp_group_idx: Int32, stage: Literal["mma", "epi"]):
        """Arrive at the OTHER warpgroup's ping-pong barrier for the given stage, without blocking.

        The release half of the alternation: this warpgroup signals that the peer may proceed.

        Args:
            warp_group_idx: Which MMA warpgroup this is (0 or 1). The barrier arrived at is the
                peer's, derived from this index.
            stage: ``"mma"`` or ``"epi"``.

        Returns:
            None.
        """
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier_arrive(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )
