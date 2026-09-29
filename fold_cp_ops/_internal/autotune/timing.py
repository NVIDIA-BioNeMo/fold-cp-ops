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

# Adapted from https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py
# Copyright (C) 2025, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""How a candidate is timed -- and why the collective case cannot use the usual timer.

CLAUDE.md states the rule this module exists to make unbreakable: a kernel containing an in-kernel
or NCCL collective must be timed with **fixed iterations**, CUDA events, and an ``all_reduce(MAX)``
for the slowest PE. An *adaptive* timer -- one that decides how many repetitions to run from a first
estimate, which is what ``triton.testing.do_bench`` and ``bench_timing``'s ``mode="device"`` both do
-- gives different rep counts on different ranks. Ranks then execute different numbers of
collectives and **desynchronize**: the fast rank exits its loop while the slow one is still inside a
collective waiting for it.

The upstream harness defaulted to ``triton.testing.do_bench``. That is not a tuning choice, it is
the desync bug, wired in as the default for every kernel including the distributed ones.

So the policy is not a parameter here; it is derived from whether a collective is present:

* **local** -- CUDA events, fixed iterations, no reduction. Same numbers as before.
* **collective** -- CUDA events, fixed iterations, ``reduce="max"``. The adaptive mode is not
  reachable: :meth:`TimingPolicy.measure` refuses it rather than accepting and ignoring it, because
  a silently-downgraded request is how the wrong timer gets used for a year.

**The layering, and why it is no longer a caveat.** The repo's tested timing primitive is
:func:`fold_cp_ops._internal.bench_timing.benchmark_single` -- CUDA events, L2 flush,
``all_reduce(MAX)`` for the slowest PE. It used to live under ``benchmark/``, which
``pyproject.toml`` does not package, so the autotuner's REQUIRED timer vanished on
``pip install`` and this module had to import it lazily and raise a message naming the fix. It now
sits inside the package and is imported plainly; ``benchmark/distributed/bench_utils.py`` re-exports
the same objects so benchmark authors keep one name. Hand-rolling a replacement stays forbidden: a
``time.perf_counter`` loop around an async launch measures host DISPATCH, not device time, and
yields "bandwidths" above the physical link.
"""

import os
from typing import Any, Callable, Optional

from fold_cp_ops._internal import bench_timing

#: Fixed measurement geometry. Fixed -- not adaptive -- because a per-rank repetition count
#: desynchronizes collectives (see the module docstring). These are the same numbers the perf gates
#: use, so an autotuned pick and a pinned measurement are directly comparable.
ROUNDS, WARMUP, ITERS = 15, 5, 10


def _default_measure(fn, *, rounds, warmup, iters, reduce, mode="event"):
    """The repo's one tested timing primitive, adapted to the tuner's keyword interface.

    Purpose
        Names the single measurement path so that "the autotuner and the perf gates measure the
        same way" is a fact about one call site rather than a convention.

    Args:
        fn: Zero-argument callable issuing exactly one kernel invocation. Must not synchronize.
        rounds: Timed rounds; the median over them is returned.
        warmup: Untimed rounds before the first timed one.
        iters: Back-to-back launches inside one event window, divided out afterwards, so the result
            carries no per-call Python overhead.
        reduce: ``"max"`` to take the slowest rank via ``all_reduce``, or None single-process.
            Under a collective this MUST be ``"max"`` -- a per-rank number is that rank's own view,
            and the winning config would be whoever waited least.
        mode: ``"device"`` for a local kernel, ``"event"`` under a collective. Chosen by
            :attr:`TimingPolicy.mode`, never here; defaulted to ``"event"`` only so an injected
            test double that omits it keeps working.

    Returns:
        Median milliseconds per call.
    """
    return bench_timing.benchmark_single(
        fn, mode=mode, rounds=rounds, warmup=warmup, iters=iters, reduce=reduce
    ).median_ms


def _accepts_mode(fn) -> bool:
    """Whether `fn` takes a ``mode`` keyword, so the policy knows if it may pass one.

    Args:
        fn: A measurement callable -- the packaged default or an injected double.

    Returns:
        True if `fn` declares a ``mode`` parameter or accepts arbitrary keywords. False for a
        callable whose signature cannot be read at all, which is the conservative answer: omitting
        `mode` yields the timer's own default rather than a TypeError.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins, C callables, odd wrappers
        return False
    return "mode" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class TimingPolicy:
    """Decides how a candidate config is measured, and refuses the combinations that desync.

    Args:
        collective: Whether the kernel being tuned contains a collective (in-kernel NVSHMEM, NCCL,
            or any cross-rank synchronization). When True the measurement is reduced across ranks
            with MAX and the iteration count is fixed. Get this WRONG in the False direction and a
            distributed kernel is timed per-rank with no reduction -- the numbers are each rank's
            own view and the winner is whoever waited least.
        measure: Optional injected ``(fn, *, rounds, warmup, iters, reduce) -> median_ms``. The
            default is :func:`_default_measure`, over the packaged
            ``bench_timing.benchmark_single``. Injecting one is how a test drives this without a
            GPU; it must honour the same fixed-iteration contract, or the collective case desyncs.
        warmup_ms: Milliseconds of GPU saturation before the sweep, to reach thermal steady state
            so the first config measured does not get an artificially good number. 0 disables it.
            Under a collective this is bracketed by barriers -- an unsynchronized warmup is itself
            a skew, since a rank still warming up makes its peers' collective look slow.
    """

    def __init__(
        self,
        *,
        collective: bool = False,
        measure: Optional[Callable[..., float]] = None,
        warmup_ms: int = 200,
    ):
        self.collective = collective
        self._measure = measure
        self.warmup_ms = warmup_ms

    def _resolve(self) -> Callable[..., float]:
        """Return the measurement callable -- the injected one, or the packaged default.

        Returns:
            A callable with `benchmark_single`'s keyword interface. Never None and never a
            hand-rolled loop: the default is :func:`_default_measure`, which is always importable
            now that the primitive is inside the package.
        """
        return self._measure if self._measure is not None else _default_measure

    @property
    def mode(self) -> str:
        """The timing mode this policy uses: ``"device"`` when local, ``"event"`` under a collective.

        Purpose
            Derived from `collective`, never passed in, for the same reason the reduction is: the
            choice is forced by whether ranks exist, and a caller who could pass it could pass the
            desyncing one.

        Semantics
            **Local -> ``"device"``.** The event window measures the CADENCE of a launch stream, not
            the kernel. Where the host submit costs more than the kernel, the GPU idles inside the
            window and the window reports host dispatch wearing the kernel's name: measured here, a
            128x256x256 GEMM read 27.3 us of "kernel time" against 7.2 us of device time. Every
            candidate at such a shape then measures the same ~22 us host floor, so the sweep is
            comparing noise and its winner is arbitrary. Observed directly at
            ``(1001, 2048, 2048, 1)``: the tuner picked ``(128,160) c(2,1)`` on one run and
            ``(128,192) c(1,2)`` on the next, 1.11x and 1.25x worse than the best-known config.

            **Collective -> ``"event"``.** Unchanged, and NOT negotiable. ``mode="device"`` decides
            its repetition count adaptively from a first estimate, so ranks run different numbers of
            collectives and desynchronize -- the fast rank leaves its loop while the slow one is
            still inside a collective waiting for it. This is the bug the module exists to prevent.

            The device mode is safe locally for exactly the reason it is unsafe distributed: with one
            rank there is nothing to stay in lockstep with.

        Returns:
            ``"event"`` or ``"device"``.
        """
        return "event" if self.collective else "device"

    def measure(self, fn: Callable[[], Any], *, mode: Optional[str] = None) -> float:
        """Time one candidate, in milliseconds.

        Args:
            fn: A zero-argument callable issuing exactly ONE kernel invocation. It must not
                synchronize -- the timer owns the events and the L2 flush, and an inner
                synchronize would serialize the very overlap being measured.
            mode: Normally None, meaning "use :attr:`mode`". Kept only so a caller who explicitly
                asks for the WRONG mode is refused rather than silently given the right one.

        Returns:
            The median milliseconds per call. Locally this is pure device time, directly comparable
            with a perf-gate pin (the gates measure the same way). Under a collective it is already
            the slowest rank's value, so it is comparable across ranks.

        Raises:
            ValueError: If `mode` is given and disagrees with :attr:`mode`. Under a collective that
                request is the desync bug asked for by name; locally it is a request to time the
                host floor instead of the kernel. Refused either way, because
                accepting-then-ignoring is how the wrong timer survives review.
        """
        if mode is not None and mode != self.mode:
            raise ValueError(
                f"this policy measures with mode={self.mode!r}; got {mode!r}. "
                + (
                    "Under a collective an adaptive timer picks its repetition count per rank, so "
                    "ranks run different numbers of collectives and desynchronize."
                    if self.collective
                    else "Locally, mode='event' measures the cadence of a launch stream, which at a "
                    "launch-bound shape is the ~22 us host submit floor rather than the kernel -- "
                    "every candidate then reads the same number and the sweep picks noise."
                )
                + " Refused rather than downgraded so the request is visible."
            )
        measure_fn = self._resolve()
        kwargs = dict(
            rounds=ROUNDS,
            warmup=WARMUP,
            iters=ITERS,
            reduce="max" if self.collective else None,
        )
        # `mode` is passed only to a callable that accepts it. An injected double predates this
        # parameter, and the injection contract is "honour the fixed-iteration semantics", not
        # "match this signature" -- so adding a required keyword would break every existing double
        # with a TypeError that says nothing about what changed. A double that omits `mode` gets the
        # event path, which is what it always got.
        if _accepts_mode(measure_fn):
            kwargs["mode"] = self.mode
        return measure_fn(fn, **kwargs)

    def thermal_warmup(self, barrier: Optional[Callable[[], None]] = None) -> None:
        """Saturate the GPU so the sweep starts at thermal steady state.

        Purpose
            Without it the first config benchmarked runs on a cool, un-throttled part and wins on
            temperature rather than on merit. That is a real effect on a sustained-clock part and it
            biases the whole sweep toward whichever config was measured first.

        Semantics
            Runs a dense matmul loop for `warmup_ms`. When a `barrier` is supplied it is called
            before AND after, so no rank is still warming up while another is timing -- an
            unsynchronized warmup converts a thermal correction into a measurement skew.

        Args:
            barrier: Optional zero-argument rank barrier. Pass `Consensus.barrier` under a
                collective; omit it single-process.

        Returns:
            None.
        """
        if self.warmup_ms <= 0:
            return
        import time

        import torch

        if barrier is not None:
            barrier()
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        deadline = time.time() + self.warmup_ms / 1000.0
        while time.time() < deadline:
            for _ in range(50):
                a = a @ a
            torch.cuda.synchronize()
        del a
        if barrier is not None:
            barrier()

    @staticmethod
    def verbose() -> bool:
        """Whether to print per-config timings. Reads ``CPO_AUTOTUNE_VERBOSE``."""
        return os.environ.get("CPO_AUTOTUNE_VERBOSE", "0") == "1"
