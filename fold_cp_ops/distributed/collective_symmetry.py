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

"""Make collective symmetry STRUCTURAL, so a one-rank divergence cannot become an all-rank hang.

Purpose
    Every NVSHMEM call and every ``torch.distributed`` collective is collective: if one rank takes a
    different path -- skips, raises, OOMs, returns early -- the others block until a watchdog aborts
    them. This module is the single place that converts a LOCAL, possibly-divergent decision into a
    GLOBAL, provably-uniform one, so that ``torchrun ... -m pytest tests/distributed/`` just works.

Why this exists as a module rather than a convention
    The upstream already gets this right in exactly one place -- the FusedTriMul build in
    ``tests/distributed/test_fused_trimul.py`` reduces a local failure flag with ``all_reduce(MAX)``
    and then skips-or-raises on every rank -- and the comments there name the two anti-patterns it
    is avoiding. But the pattern is open-coded, it is applied at one call site out of many, and the
    upstream's own ``tests/distributed/_topology.py`` docstring points at a file that does not exist
    for the canonical copy. A correct pattern that must be remembered is one that will be forgotten:
    the measured cost of forgetting it here is a 30-minute hang whose traceback names an unrelated
    test N tests later. So the pattern becomes an object with four entry points, and the call sites
    become one line each.

Semantics
    The core primitive is `CollectiveGate.agree_bool`: a `all_reduce` of a 0/1 flag that EVERY rank
    reaches on EVERY path. Everything else is built on it:

    * `guard` -- run a block; whatever it raises, reduce the failure flag FIRST and only then
      re-raise, so no rank can bypass the reduction on its way out. Ranks that succeeded raise
      `PeerFailure`; the rank that failed raises `CollectiveFailure` wrapping its own exception.
    * `should_skip` -- reduce a skip DECISION and return the agreed answer, so callers skip
      together or not at all. Returns a reason rather than calling ``pytest.skip``, which keeps
      this module free of a pytest import and testable without one.
    * `barrier` -- a bounded barrier that RAISES on timeout. Contrast the upstream's cleanup
      barrier, which warns and then proceeds INTO a collective ``nvshmem`` finalize, converting a
      merely-slow peer into a corrupted one.

    **What this module cannot do.** It cannot rescue a rank that has already died or wandered into
    a collective alone -- by then the symmetry is gone. It only guarantees that decisions routed
    THROUGH it are uniform. A code path that skips without asking is still a defect; this module
    makes the correct spelling a one-liner, not a design exercise.

Input requirements
    An initialized ``torch.distributed`` process group. With the NCCL backend the reduction tensor
    must live on the calling rank's CUDA device, so `device` must name that device -- passing a CPU
    device to a NCCL group raises inside the collective rather than here. Every method is
    COLLECTIVE and must be reached by every rank in the group, in the same order; that requirement
    is the thing being enforced, so violating it inside this module's own callers is the one
    failure it cannot detect.

Raises
    `CollectiveFailure`, `PeerFailure`, `BarrierTimeout` -- all subclasses of `CollectiveError`, so
    a caller that only wants "did the group diverge" catches one type.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import time
from typing import Optional

import torch
import torch.distributed as dist

#: Default seconds a `CollectiveGate.barrier` waits before RAISING.
#:
#: The bound is squeezed between TWO constraints and must satisfy both, which is easy to get wrong
#: because only the first is obvious:
#:
#: 1. **Well ABOVE the slowest legitimate cell.** The upstream used 30 s, and a 30 s bound on a suite
#:    whose large-N cells legitimately take longer converts a slow peer into a corrupted one -- the
#:    cleanup barrier times out, warns, and falls straight through into a collective finalize. A
#:    bound exists to catch a DEAD peer, not a slow one. The worst measured cold compile in this repo
#:    is 228 s (see the compile-time defect in CLAUDE.md), so anything under ~300 s is unsafe.
#:
#: 2. **Strictly BELOW NCCL's own watchdog**, which is what this used to get wrong. Torch's
#:    `WorkNCCL` watchdog defaults to 600000 ms, and this default was ALSO 600 s -- so the two tied
#:    and it became a race. Measured: NCCL won, aborting the process with a generic
#:    `Watchdog caught collective operation timeout` instead of letting `BarrierTimeout` reach
#:    `pytest_runtest_teardown`, which exits with the failing node id and names the usual causes
#:    (a per-rank skip, an unguarded exception). **The instrument that DIAGNOSES lost a coin-flip to
#:    the one that merely aborts**, and a tie means it loses half the time.
#:
#: 480 s clears the slowest cell by >2x and leaves NCCL 120 s of margin. Raise
#: `CPO_BARRIER_TIMEOUT_S` only together with the process group's own timeout, never alone -- a value
#: at or above the NCCL watchdog silently restores the race this constant exists to avoid.
DEFAULT_BARRIER_TIMEOUT_S = float(os.environ.get("CPO_BARRIER_TIMEOUT_S", "480"))

#: Seconds between polls while waiting on an async barrier. Small enough to keep teardown snappy,
#: large enough that the poll loop costs nothing against a bound of several minutes.
_POLL_S = 0.05


class CollectiveError(RuntimeError):
    """Base for every divergence this module detects. Catch this to mean 'the group disagreed'."""


class CollectiveFailure(CollectiveError):
    """Raised on the rank whose own guarded block failed, wrapping the original exception."""


class PeerFailure(CollectiveError):
    """Raised on ranks whose guarded block SUCCEEDED, because a peer's did not.

    Carries no traceback from the originating rank -- it cannot, the exception never crossed the
    wire. The originating rank's log holds the real error; this exception exists so the successful
    ranks fail at the SAME point instead of walking into the next collective alone.
    """


class BarrierTimeout(CollectiveError):
    """Raised when a bounded barrier did not complete, naming the bound that was exceeded."""


class CollectiveGate:
    """One object per process group: turns local decisions into group-uniform ones.

    Args:
        group: The process group to reduce over. ``None`` means the default group. Every method is
            collective over exactly this group; mixing gates over different groups in one code path
            is how a deadlock is written, so prefer one gate per mesh and pass it around.
        device: Device for reduction tensors. ``None`` derives ``cuda:<local_rank>`` when CUDA is
            available, else CPU. Must be the calling rank's own device under NCCL -- a mismatched
            device raises inside the collective, not here.
        timeout_s: Default bound for `barrier`. See `DEFAULT_BARRIER_TIMEOUT_S` for why it is large.

    Raises:
        RuntimeError: If the process group is not initialized, checked eagerly so the failure names
            the real cause instead of surfacing as a confusing collective error later.
    """

    def __init__(self, group=None, device=None, timeout_s: float = DEFAULT_BARRIER_TIMEOUT_S):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "CollectiveGate needs an initialized torch.distributed process group; call "
                "init_process_group first. Constructing it earlier hides the real error behind a "
                "collective failure several frames later."
            )
        self.group = group
        self.timeout_s = timeout_s
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)

    @property
    def rank(self) -> int:
        """This rank's index within the gate's group."""
        return dist.get_rank(self.group)

    @property
    def world_size(self) -> int:
        """Number of ranks in the gate's group."""
        return dist.get_world_size(self.group)

    def agree_bool(self, local: bool, *, op: str = "any") -> bool:
        """Reduce a local flag to a group-uniform answer. COLLECTIVE.

        This is the primitive the rest of the module is built on, and the one rule for using it is:
        **every rank must reach this call on every path.** A rank that returns early before it is
        the defect this module exists to prevent.

        Args:
            local: This rank's flag.
            op: ``"any"`` -> True if ANY rank passed True (reduced with MAX). ``"all"`` -> True only
                if EVERY rank did (reduced with MIN). Anything else raises `ValueError` rather than
                silently picking one, because the two differ exactly when it matters.

        Returns:
            The reduced flag, identical on every rank.

        Raises:
            ValueError: If `op` is neither ``"any"`` nor ``"all"``.
        """
        if op not in ("any", "all"):
            raise ValueError(f"op must be 'any' or 'all'; got {op!r}")
        reduce_op = dist.ReduceOp.MAX if op == "any" else dist.ReduceOp.MIN
        t = torch.tensor([1.0 if local else 0.0], device=self.device)
        dist.all_reduce(t, op=reduce_op, group=self.group)
        return bool(t.item() > 0)

    def barrier(self, timeout_s: Optional[float] = None) -> None:
        """A barrier that RAISES on timeout instead of warning and continuing. COLLECTIVE.

        Args:
            timeout_s: Bound in seconds; ``None`` uses the gate's default. A bound of zero or less
                raises `ValueError` -- an unbounded barrier is the failure mode this replaces, so
                it may not be requested by passing 0.

        Returns:
            None, once every rank has arrived.

        Raises:
            BarrierTimeout: If the bound elapses. The caller MUST NOT proceed into another
                collective after catching this: by then the group is desynchronized and the only
                safe actions are reporting and exiting.
            ValueError: If `timeout_s` is not positive.
        """
        bound = self.timeout_s if timeout_s is None else timeout_s
        if bound <= 0:
            raise ValueError(f"barrier timeout must be positive; got {bound}")
        if os.environ.get("CPO_BARRIER_WAIT", "poll") == "sync":
            # `main`'s EXACT form. This arm answers the one question the poll/wait A/B cannot --
            # does the form `main` actually uses wedge here -- because both of those hang off the
            # `async_op=True` below and are spellings we invented.
            #
            # COST, stated because it is real: no Work object means no bound, no `BarrierTimeout`,
            # and no `NAMED DESYNC:`. A genuine desync hangs until the harness kills it, exactly as
            # on `main`. Accepted for the duration of the arm; NOT a candidate for the shipped
            # default until the measurement says the async form is the defect.
            #
            # `self.device.index` is the live CUDA ordinal (`__init__` defaults device to
            # `torch.device("cuda", torch.cuda.current_device())`), i.e. the same integer `main`
            # gets from `_env_local_rank()`. On CPU `.index` is None, so branch as `main` does
            # rather than passing `[None]`.
            if self.device.type == "cuda":
                dist.barrier(group=self.group, device_ids=[self.device.index])
            else:
                dist.barrier(group=self.group)
            return
        work = dist.barrier(group=self.group, async_op=True)
        if work is None:
            # Some backends complete the barrier inline and return nothing rather than a Work. That
            # is already-arrived, not a failure -- but it also means the bound was not applied, so
            # do NOT pretend otherwise by looping on a None.
            return
        if os.environ.get("CPO_BARRIER_WAIT", "poll") == "wait":
            # `Work.wait(timeout=...)` is torch's OWN bounded wait, and it DRIVES the work to
            # completion rather than asking whether it happened. The poll branch below does not: it
            # calls `is_completed()`, which queries state, and relies on progress occurring as a side
            # effect of something else. `main`'s conftest sidesteps the question entirely by calling
            # a SYNCHRONOUS `torch.distributed.barrier(device_ids=[...])`.
            #
            # Suspected, NOT yet demonstrated, which is why this is env-gated rather than swapped:
            # in a 12-cell wedge the 480 s bound had not raised at 540 s of polling and NCCL's 600 s
            # watchdog spoke first -- the exact inversion the bound exists to prevent. A poll loop
            # that never reaches its own deadline check is consistent with `is_completed()` not
            # returning, and inconsistent with a loop that is merely slow.
            #
            # Graded on the A/B rig against the poll branch, both arms concurrent from one file, per
            # the race finding: a single run cannot distinguish these when the failure is a race.
            # `Work.wait(timeout=)` RAISES on expiry -- it does not return False. MEASURED:
            # `torch.distributed.DistBackendError` after 5.6s against a 5s bound
            # (`w8plan/wait_raises_probe.py`), message "Watchdog caught collective operation
            # timeout: WorkNCCL(...)". The earlier `if not work.wait(...)` therefore had an
            # UNREACHABLE raise: torch threw first, the teardown hook's `except BarrierTimeout` did
            # not match, the generic handler warned, and `_group_broken` was never set -- so the
            # session walked into the next collective on a timed-out group, which is exactly what
            # this class exists to prevent. The arm had neither a bound nor a gate.
            #
            # Caught broadly rather than by name: the concrete type is backend-dependent (NCCL gave
            # DistBackendError here) and a miss reinstates the silent-warning path. The `not
            # completed` branch is kept because the docstring's stated contract is a bool return,
            # and a future torch that honours it must still raise BarrierTimeout.
            try:
                completed = work.wait(timeout=datetime.timedelta(seconds=bound))
            except Exception as e:  # noqa: BLE001 -- see above; a miss silently removes the bound
                raise BarrierTimeout(
                    f"rank {self.rank} waited {bound:.0f}s at a collective barrier (wait) and at "
                    f"least one peer never arrived ({type(e).__name__}). The group is now "
                    f"desynchronized -- do NOT enter another collective."
                ) from e
            if not completed:
                raise BarrierTimeout(
                    f"rank {self.rank} waited {bound:.0f}s at a collective barrier (wait) and at "
                    f"least one peer never arrived. The group is now desynchronized -- do NOT enter "
                    f"another collective."
                )
            return

        deadline = time.monotonic() + bound
        while not work.is_completed():
            if time.monotonic() > deadline:
                raise BarrierTimeout(
                    f"rank {self.rank} waited {bound:.0f}s at a collective barrier and at least one "
                    f"peer never arrived. The group is now desynchronized -- do NOT enter another "
                    f"collective. A peer has died, hung, or taken a divergent path (a per-rank skip "
                    f"or an unguarded exception are the usual causes)."
                )
            time.sleep(_POLL_S)

    @contextlib.contextmanager
    def guard(self, what: str):
        """Run a block so that a failure on ANY rank becomes a failure on EVERY rank. COLLECTIVE.

        Semantics
            The block's exception is caught LOCALLY and held. Then -- unconditionally, on every rank,
            whether or not it failed -- the failure flag is reduced. Only after the reduction does
            anyone raise. That ordering is the whole point: an implementation that re-raised
            immediately would let the failing rank exit while its peers block in the reduction, which
            is the deadlock this replaces.

        Args:
            what: Short label for the guarded region, e.g. ``"build"`` or ``"oracle"``. Appears in
                both exception messages and is the only clue about WHERE the group diverged, so name
                the operation, not the file.

        Yields:
            None.

        Raises:
            CollectiveFailure: On the rank whose block raised, chained (``from``) to the original.
            PeerFailure: On ranks whose block succeeded while another's did not.
        """
        local_exc: Optional[BaseException] = None
        try:
            yield
        except BaseException as exc:  # noqa: BLE001 - re-raised below, after the reduction
            local_exc = exc
        failed_anywhere = self.agree_bool(local_exc is not None, op="any")
        if not failed_anywhere:
            return
        if local_exc is not None:
            raise CollectiveFailure(
                f"[{what}] failed on rank {self.rank} (surfaced collectively so no rank proceeds "
                f"alone): {type(local_exc).__name__}: {local_exc}"
            ) from local_exc
        raise PeerFailure(
            f"[{what}] succeeded on rank {self.rank} but FAILED on at least one peer, so this rank "
            f"stops here rather than entering the next collective alone. The failing rank's log "
            f"holds the original traceback."
        )

    def should_skip(self, local_reason: Optional[str]) -> Optional[str]:
        """Agree on whether to skip, so ranks skip together or not at all. COLLECTIVE.

        Returns a reason instead of raising ``pytest.skip`` so this module needs no pytest import
        and can be unit-tested without one; the pytest glue lives in ``tests/distributed/conftest.py``.

        Args:
            local_reason: This rank's reason to skip, or ``None`` to proceed. Only rank-DIVERGENT
                predicates need this -- a rank-invariant one (a job property every rank computes
                identically, such as the cp peer table) is already uniform and a plain skip is safe.
                Routing an invariant predicate through here is harmless but costs a collective.

        Returns:
            The agreed reason if ANY rank wanted to skip, else ``None``. The returned text is this
            rank's own reason when it had one, otherwise a note that a peer asked -- so a log never
            claims a condition the local rank did not actually observe.
        """
        skip = self.agree_bool(local_reason is not None, op="any")
        if not skip:
            return None
        return local_reason or "a peer rank requested a skip (this rank's own predicate passed)"
