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

# Copyright (c) 2025-2026, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

from typing import Optional
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate, and_, dsl_user_op
from cutlass.pipeline import MbarrierArray, CooperativeGroup, PipelineOp
from cutlass.pipeline import PipelineState, PipelineUserType
from cutlass.pipeline import Agent, agent_sync
from cutlass.pipeline import NamedBarrier as NamedBarrierOg
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineCpAsync as PipelineCpAsyncOg
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg
from cutlass.pipeline import PipelineUmmaAsync as PipelineUmmaAsyncOg
from cutlass.pipeline import PipelineAsyncUmma as PipelineAsyncUmmaOg


# ── Shared helpers ───────────────────────────────────────────────────────────


def _override_create(parent_cls, child_cls):
    """Create a static factory that constructs parent_cls then re-classes to child_cls."""

    @staticmethod
    def create(*args, **kwargs):
        """Build the parent pipeline, then re-class the instance to this subclass.

        Args:
            *args: Forwarded to the parent's ``create``.
            **kwargs: Forwarded to the parent's ``create``.

        Returns:
            The parent's object, re-classed in place. Re-classing rather than subclass-constructing
            is what lets these thin overrides reuse the parent's whole construction path; the
            dataclass is frozen, so the assignment goes through ``object.__setattr__``.
        """
        obj = parent_cls.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", child_cls)
        return obj

    return create


def _make_state(index: Int32, phase: Int32) -> PipelineState:
    """Construct a PipelineState from index and phase (count/stages unused by callers)."""
    return PipelineState(stages=0, count=Int32(0), index=index, phase=phase)


def _call_with_elect_one(parent_method, self, state, elect_one, syncwarp, loc, ip):
    """Optionally wrap a parent pipeline method call in sync_warp + elect_one."""
    if const_expr(elect_one):
        if const_expr(syncwarp):
            cute.arch.sync_warp()
        with cute.arch.elect_one():
            parent_method(self, state, loc=loc, ip=ip)
    else:
        parent_method(self, state, loc=loc, ip=ip)


# ── Pipeline state ──────────────────────────────────────────────────────────


class PipelineStateWAdvance(PipelineState):
    """A ``PipelineState`` that can jump forward by N stages, not just one.

    The stock state advances one stage at a time. A persistent kernel needs to skip -- when a work
    tile is shorter than the pipeline depth, the next tile starts several stages on -- and doing
    that by looping ``advance()`` costs an instruction per skipped stage in the inner loop.

    The phase bit is what makes this non-trivial: it flips once per wrap of the stage ring, so a
    multi-stage jump has to flip it once per *crossing*, which is the integer quotient rather than
    a single toggle.
    """

    @dsl_user_op
    def advance_iters(self, num_iterations: Int32, *, loc=None, ip=None):
        """Advance the state by ``num_iterations`` stages, flipping the phase once per wrap.

        Args:
            num_iterations: How many stages to skip. Must be non-negative -- the modulo and the
                crossing count both assume it, and a negative value produces an index outside the
                ring rather than moving backwards.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None; the state is mutated in place.
        """
        self._count += Int32(num_iterations)
        new_index = self._index + Int32(num_iterations)
        # How many times did we cross the stages boundary
        num_crossings = new_index // self.stages
        self._phase ^= num_crossings
        self._index = new_index % self.stages

    # This can be overridden by derived classes
    def __new_from_mlir_values__(self, values):
        """Rebuild the state inside the kernel from its three marshalled scalars.

        Args:
            values: ``(count, index, phase)`` as MLIR values, in that order. ``stages`` is a
                compile-time constant taken from ``self``, not from the values.

        Returns:
            A new ``PipelineStateWAdvance``. Overridable by subclasses that add state.
        """
        return PipelineStateWAdvance(
            self.stages, Int32(values[0]), Int32(values[1]), Int32(values[2])
        )


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        return PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(1))
    elif type is PipelineUserType.Consumer:
        return PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


# ── Mixin: _w_index / _w_index_phase variants ───────────────────────────────


class _PipelineIndexPhaseMixin:
    """Mixin providing _w_index_phase / _w_index methods that delegate to PipelineState-based parents."""

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        """Acquire a producer stage named by an explicit ``(index, phase)``.

        Args:
            index: Stage index within the pipeline ring.
            phase: Phase bit for that stage. It must match what the ring is actually on -- a wrong
                phase waits on a barrier that has already flipped, which hangs.
            try_acquire_token: Optional token from a prior non-blocking try.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        state = _make_state(index, phase)
        self.producer_acquire(state, try_acquire_token, loc=loc, ip=ip)

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        """Commit a producer stage named by an explicit index.

        Args:
            index: Stage index. No phase is needed -- committing signals the full barrier, which
                does not read the phase.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        state = _make_state(index, Int32(0))
        self.producer_commit(state, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        """Wait on a consumer stage named by an explicit ``(index, phase)``.

        Args:
            index: Stage index within the ring.
            phase: Phase bit for that stage; a wrong one hangs, as above.
            try_wait_token: Optional token from a prior non-blocking try.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        state = _make_state(index, phase)
        self.consumer_wait(state, try_wait_token, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        """Release a consumer stage named by an explicit index.

        Args:
            index: Stage index. No phase is needed for the same reason as the commit path.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        state = _make_state(index, Int32(0))
        self.consumer_release(state, loc=loc, ip=ip)


# ── NamedBarrier ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NamedBarrier(NamedBarrierOg):
    """A named barrier addressable by an OFFSET from its base id.

    Hardware named barriers are numbered, and a kernel that needs one per pipeline stage would
    otherwise have to declare a separate object per stage. The two ``_w_index`` methods add the
    index to ``barrier_id`` at the call site instead.

    Input requirement, unchecked: ``barrier_id + index`` must stay within the hardware's 16 named
    barriers, and must not collide with another barrier the kernel uses. Barrier 0 is reserved for
    ``__syncthreads``.
    """

    create = _override_create(NamedBarrierOg, None)  # patched below

    @dsl_user_op
    def arrive_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        """
        The aligned flavor of arrive is used when all threads in the CTA will execute the
        same instruction. See PTX documentation.
        """
        cute.arch.barrier_arrive(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_and_wait_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        """Arrive at barrier ``barrier_id + index`` and block until every participant has.

        Args:
            index: Offset from the base barrier id.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None. Exactly ``num_threads`` threads must reach this call, or the CTA hangs.
        """
        cute.arch.barrier(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )


NamedBarrier.create = _override_create(NamedBarrierOg, NamedBarrier)


# ── PipelineAsync ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsync(_PipelineIndexPhaseMixin, PipelineAsyncOg):
    """
    PipelineAsync with optional elect_one for producer_commit and consumer_release.

    When elect_one_*=True (set at create time), only one elected thread per warp
    signals the barrier arrive. This is useful when the mask count is set to 1 per warp.

    Args (to create):
        elect_one_commit: If True, only elected thread signals producer_commit.
        syncwarp_before_commit: If True (default), issue syncwarp before elect_one.
        elect_one_release: If True, only elected thread signals consumer_release.
        syncwarp_before_release: If True (default), issue syncwarp before elect_one.
            Set syncwarp to False when threads are already converged (e.g. after wgmma wait_group).
    """

    _elect_one_commit: bool = False
    _syncwarp_before_commit: bool = True
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_commit: bool = False,
        syncwarp_before_commit: bool = True,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        """Build a ``PipelineAsync`` with the elect-one commit/release policy bound in.

        Args:
            *args: Forwarded to the parent's ``create``.
            elect_one_commit: Have a single elected thread per warp signal ``producer_commit``.
                Correct only when the barrier's arrive count is 1 per warp; with a full count the
                barrier never completes.
            syncwarp_before_commit: Issue ``sync_warp`` before electing. Leave True unless the warp
                is already converged (e.g. straight after a ``wgmma`` wait), where it is redundant.
            elect_one_release: The same for ``consumer_release``.
            syncwarp_before_release: The same for the release path.
            **kwargs: Forwarded to the parent's ``create``.

        Returns:
            The re-classed pipeline object carrying the four policy flags.
        """
        obj = PipelineAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineAsync)
        object.__setattr__(obj, "_elect_one_commit", elect_one_commit)
        object.__setattr__(obj, "_syncwarp_before_commit", syncwarp_before_commit)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def producer_commit(self, state: PipelineState, *, loc=None, ip=None):
        """Signal that this stage's data is ready, optionally from one elected thread per warp.

        Args:
            state: The producer's pipeline state, naming the stage.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        _call_with_elect_one(
            PipelineAsyncOg.producer_commit,
            self,
            state,
            self._elect_one_commit,
            self._syncwarp_before_commit,
            loc,
            ip,
        )

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        """Release this stage back to the producer, optionally from one elected thread per warp.

        Args:
            state: The consumer's pipeline state, naming the stage.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        _call_with_elect_one(
            PipelineAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )

    # _w_index variants inherited from _PipelineIndexPhaseMixin, which delegate
    # to producer_commit / consumer_release above.


# ── PipelineCpAsync ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineCpAsync(_PipelineIndexPhaseMixin, PipelineCpAsyncOg):
    """``PipelineCpAsync`` with an optional elect-one ``consumer_release``, plus index/phase entries.

    Only the release side is configurable here: a ``cp.async`` producer commits through the
    hardware's async-group mechanism rather than an explicit barrier arrive, so there is no commit
    to elect on.
    """

    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        """Build a ``PipelineCpAsync`` with the elect-one release policy bound in.

        Args:
            *args: Forwarded to the parent's ``create``.
            elect_one_release: Have one elected thread per warp signal ``consumer_release``.
                Correct only when the barrier's arrive count is 1 per warp.
            syncwarp_before_release: Issue ``sync_warp`` before electing.
            **kwargs: Forwarded to the parent's ``create``.

        Returns:
            The re-classed pipeline object.
        """
        obj = PipelineCpAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineCpAsync)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        """Release this stage, optionally from one elected thread per warp.

        Args:
            state: The consumer's pipeline state.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        _call_with_elect_one(
            PipelineCpAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )

    # _w_index variants inherited from _PipelineIndexPhaseMixin.


# ── PipelineTmaAsync ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaAsync(_PipelineIndexPhaseMixin, PipelineTmaAsyncOg):
    """Override producer_acquire to take in extra_tx_count parameter."""

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count, loc=loc, ip=ip)


PipelineTmaAsync.create = _override_create(PipelineTmaAsyncOg, PipelineTmaAsync)


# ── PipelineTmaUmma ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaUmma(_PipelineIndexPhaseMixin, PipelineTmaUmmaOg):
    """Override producer_acquire to take in extra_tx_count parameter."""

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive(
                    state.index, self.producer_mask, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive_and_expect_tx(
                    state.index, tx_count, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )


PipelineTmaUmma.create = _override_create(PipelineTmaUmmaOg, PipelineTmaUmma)


# ── PipelineUmmaAsync ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineUmmaAsync(_PipelineIndexPhaseMixin, PipelineUmmaAsyncOg):
    """``PipelineUmmaAsync`` plus the ``_w_index`` / ``_w_index_phase`` entry points.

    Adds no behaviour of its own -- the mixin's methods let a caller drive the pipeline by an
    explicit ``(index, phase)`` instead of a ``PipelineState`` object, which is what a scheduler
    that computes its own stage arithmetic needs. UMMA is SM100; unreachable from this package's
    SM90 kernels, and kept because ``pipeline.py`` is shared machinery.
    """

    pass


PipelineUmmaAsync.create = _override_create(PipelineUmmaAsyncOg, PipelineUmmaAsync)


# ── PipelineAsyncUmma ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsyncUmma(_PipelineIndexPhaseMixin, PipelineAsyncUmmaOg):
    """``PipelineAsyncUmma`` plus the ``_w_index`` / ``_w_index_phase`` entry points.

    Adds no behaviour of its own; see :class:`PipelineUmmaAsync`. Also SM100-only.
    """

    pass


PipelineAsyncUmma.create = _override_create(PipelineAsyncUmmaOg, PipelineAsyncUmma)


# ── PipelineTmaCpAsync ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaCpAsync(_PipelineIndexPhaseMixin, PipelineTmaAsyncOg):
    """
    PipelineTmaCpAsync is used for CpAsync + TMA producers and AsyncThread consumers.
    Compared to PipelineTmaAsync, producer_acquire gates the full-barrier arrive on is_tma_warp.
    """

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
        *,
        loc=None,
        ip=None,
    ):
        """Wait for the stage to be empty, then arrive on the full barrier from the TMA warp only.

        The difference from ``PipelineTmaAsync``: several warps call this (the cp.async producers
        alongside the TMA one), but the full barrier's transaction count is set for a single TMA
        arrival. Gating the arrive on ``is_tma_warp`` is what keeps the count right.

        Args:
            state: The producer's pipeline state.
            try_acquire_token: Optional token from a prior non-blocking try; when it is None or 0
                this call blocks on the empty barrier, otherwise the wait is skipped.
            is_tma_warp: Whether THIS warp is the one issuing the TMA. Every producer warp must
                pass its own value; passing True from more than one over-arrives the barrier and
                the consumer proceeds on data that has not landed.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        # This is the difference between this and PipelineTmaAsync: we could have multiple
        # warps calling this, but only 1 warp should do the arrive on the full barrier
        if_generate(
            is_tma_warp,
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_cpasync_commit(self, state: PipelineState, *, loc=None, ip=None):
        """Signal this stage's mbarrier that the producer's ``cp.async`` copies completed.

        The mbarrier is what tracks ``cp.async`` completion -- the copies are asynchronous
        and the consumer has nothing else to wait on.

        **Nothing in this tree calls it, and that is PENDING rather than dead.** Upstream it
        is driven by the ``gather_A`` / varlen ``cp.async`` producer loop
        (``main:kernels/gemm_sm90.py:1298`` and ``:1318``), which this branch has not brought
        back. **Do not remove it on a no-callers finding alone.**

        Args:
            state: Producer state naming the stage whose barrier to signal.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        cute.arch.cp_async_mbarrier_arrive_noinc(
            self.producer_get_barrier(state, loc=loc, ip=ip), loc=loc, ip=ip
        )


PipelineTmaCpAsync.create = _override_create(PipelineTmaAsyncOg, PipelineTmaCpAsync)


# ── MbarrierArrayWDropCount ─────────────────────────────────────────────────


class MbarrierArrayWDropCount(MbarrierArray):
    """An mbarrier array whose arrive count is reduced by a **runtime** drop count.

    The stock array fixes its arrive count from the cooperative group's size at trace time. That is
    wrong whenever some participants may exit before arriving -- a producer warpgroup that drains
    early, or a specialization active only for part of the grid. Subtracting their count leaves a
    barrier the survivors can still complete.

    The drop count is a runtime value, so it crosses the host -> kernel boundary; the two protocol
    methods below carry it alongside the storage pointer.
    """

    @dsl_user_op
    def __init__(
        self,
        barrier_storage: cute.Pointer,
        num_stages: int,
        agent: tuple[PipelineOp, CooperativeGroup],
        tx_count: int = 0,
        drop_count: Optional[Int32] = None,
        *,
        loc=None,
        ip=None,
    ) -> None:
        """Initialize an mbarrier array whose arrive count is reduced by a runtime drop count.

        The reason this class exists: a warpgroup that exits early (a drained producer, a
        specialization that finishes ahead) must not be waited for. Subtracting its arrivals from
        the barrier's count at construction is how the remaining participants still complete.

        Args:
            barrier_storage: SMEM pointer to the barrier array. Must be 8-byte aligned and large
                enough for ``num_stages`` barriers.
            num_stages: Number of barriers in the array. Must be positive.
            agent: ``(op_type, cooperative_group)``; the group's size is the base arrive count.
            tx_count: Expected transaction bytes for a TMA-load barrier. Must be non-negative for
                ``PipelineOp.TmaLoad``.
            drop_count: How many arrivals to subtract, or None for the full count. May be a runtime
                ``Int32``. **Unchecked against the group size** -- a drop count that meets or
                exceeds it leaves a barrier no one can complete, which is a hang, not an error.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None. The barriers are initialized here, in the constructor, so the object is usable
            immediately.

        Raises:
            ValueError: If ``num_stages`` or the arrive count is not positive, or if ``tx_count``
                is negative for a TMA op.
        """
        self.barrier_storage = barrier_storage
        self.tx_count = tx_count
        self.num_stages = num_stages
        self.op_type, self.cg = agent
        self.arrive_count = self.cg.size
        self.drop_count = drop_count

        if self.num_stages <= 0:
            raise ValueError("Error: Mbarrier stage count must be greater than 0.")
        if self.arrive_count <= 0:
            raise ValueError("Error: Mbarrier arrive count must be greater than 0.")
        if self.op_type is PipelineOp.TmaLoad and self.tx_count < 0:
            raise ValueError("Error: Mbarrier tx count must not be less than 0 for TMA ops.")

        if const_expr(drop_count is not None):
            self.arrive_count = self.arrive_count - drop_count

        # Store mbarrier base pointer
        self.mbarrier_base = self.barrier_storage

        # Mbarrier initialization in constructor
        self.mbarrier_init(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        """Flatten to the two values that vary at runtime.

        Returns:
            ``[barrier_storage, drop_count]``. Everything else -- stage count, op type, cooperative
            group, transaction count -- is a compile-time constant read from ``self`` on rebuild.
        """
        return [self.barrier_storage, self.drop_count]

    def __new_from_mlir_values__(self, values):
        """Rebuild the array inside the kernel from the two marshalled values.

        Args:
            values: ``[barrier_storage, drop_count]``, as extracted.

        Returns:
            A new ``MbarrierArrayWDropCount``.

        Note:
            The constructor re-runs ``mbarrier_init``. That is correct on the device side, where
            this rebuild happens once per kernel with the barriers not yet in use -- but it means
            the object must not be reconstructed mid-flight, which would reset barriers other warps
            are waiting on.
        """
        return MbarrierArrayWDropCount(
            values[0], self.num_stages, (self.op_type, self.cg), self.tx_count, values[1]
        )


# ── PipelineTmaCpAsyncUmma ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaCpAsyncUmma(PipelineTmaUmmaOg):
    """
    PipelineTmaCpAsync is used for CpAsync + TMA producers and UMMA consumers
    (e.g. Blackwell mainloops)
    """

    @dsl_user_op
    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        mcast_mode_mn: tuple[int, int] = (1, 1),
        defer_sync: bool = False,
        producer_drop_count: Optional[Int32] = None,
        loc=None,
        ip=None,
    ):
        """Creates and initializes a new PipelineTmaUmma instance.

        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: int
        :param producer_group: CooperativeGroup for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: CooperativeGroup for the consumer agent
        :type consumer_group: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        :param barrier_storage: Pointer to the shared memory address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer, optional
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout, optional
        :param mcast_mode_mn: Tuple specifying multicast modes for m and n dimensions (each 0 or 1)
        :type mcast_mode_mn: tuple[int, int], optional
        :raises ValueError: If barrier_storage is not a cute.Pointer instance
        :return: A new PipelineTmaUmma instance configured with the provided parameters
        :rtype: PipelineTmaUmma
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise TypeError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.TCGen05Mma

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = MbarrierArrayWDropCount(
            barrier_storage.align(min_align=8),
            num_stages,
            producer,
            tx_count,
            drop_count=producer_drop_count,
            loc=loc,
            ip=ip,
        )
        sync_object_empty = PipelineTmaUmmaOg._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages,
            num_stages,
            consumer,
            loc=loc,
            ip=ip,
        )

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, loc=loc, ip=ip) == 1:
            # No mcast mask if not using clusters
            producer_mask = None
            # All threadblocks are leaders if not using clusters
            is_leader_cta = True
        else:
            producer_mask = PipelineTmaUmmaOg._compute_mcast_arrival_mask(
                cta_layout_vmnk, mcast_mode_mn, loc=loc, ip=ip
            )
            is_leader_cta = PipelineTmaUmmaOg._compute_is_leader_cta(
                cta_layout_vmnk, loc=loc, ip=ip
            )

        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0], loc=loc, ip=ip) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )

        consumer_mask = producer_mask

        if not defer_sync:
            cute.arch.mbarrier_init_fence()
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, loc=loc, ip=ip) == 1:
                agent_sync(Agent.ThreadBlock)
            else:
                agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)

        return PipelineTmaCpAsyncUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cta_group,
        )

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the
        transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        # This is the difference between this and PipelineTmaAsync: we could have multiple
        # warps calling this, but only 1 warp should do the arrive on the full barrier
        if_generate(
            and_(self.is_leader_cta, is_tma_warp),
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_cpasync_commit(self, state: PipelineState, *, loc=None, ip=None):
        """Signal this stage's mbarrier that the producer's ``cp.async`` copies completed.

        The mbarrier is what tracks ``cp.async`` completion -- the copies are asynchronous
        and the consumer has nothing else to wait on.

        **Nothing in this tree calls it, and that is PENDING rather than dead.** Upstream it
        is driven by the ``gather_A`` / varlen ``cp.async`` producer loop
        (``main:kernels/gemm_sm90.py:1298`` and ``:1318``), which this branch has not brought
        back. **Do not remove it on a no-callers finding alone.**

        Args:
            state: Producer state naming the stage whose barrier to signal.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        cute.arch.cp_async_mbarrier_arrive_noinc(
            self.producer_get_barrier(state, loc=loc, ip=ip), loc=loc, ip=ip
        )
