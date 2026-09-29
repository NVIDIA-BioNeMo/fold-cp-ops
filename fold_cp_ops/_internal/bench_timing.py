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

"""Reusable CUDA-event benchmarking helpers for local (single-GPU) and distributed (multi-rank) modules.

Ported / distilled from the upstream CP project's ``benchmark_modules_cuda_events.py`` (the CLI
harness: stream-pinned ``torch.cuda.Event`` timing + warmup-drift guard) and the
fold_cp_ops ``benchmark/trimul_autotune/bench_trimul.py`` drift-cancelled paired-median
idiom (per-round back-to-back timing + median-of-ratios). Both anti-patterns are
stripped (see ``T0_4_FINDINGS.md``): this is an *importable library*, not a
source-loading CLI; it carries no ``DistributedManager`` import, no per-shape
fixture protocol, and no JSON-record machinery.

Bench discipline baked in (the four project requirements):

  (a) Fold one-time host/setup work ONCE, then loop ONLY the kernel/module launch.
      The caller builds inputs / weights / warm-compiles BEFORE handing a
      zero-arg ``fn`` to the timer; every timed call is just ``fn()``. Inside one
      timed window we record a single event pair around ``iters`` back-to-back
      ``fn()`` launches and divide by ``iters`` — so the result carries no
      per-call Python/setup overhead and a profiler attached to the window sees
      only the real kernel(s).

  (b) Paired / interleaved timing (``benchmark_paired``): each round times every
      target (and the optional baseline) back-to-back, and we report the MEDIAN
      over rounds of each per-round ``target/baseline`` RATIO. A common-mode GPU
      clock drift scales every target in a round by the same factor, so it
      cancels in the ratio (the raw ``ms`` still drifts; the ratio does not).

  (c) Warmup + drain. ``warmup`` untimed rounds run the whole back-to-back window
      first (burns in JIT caches + settles clocks); every timed window brackets
      its launches with ``torch.cuda.synchronize(device)`` so the start event is
      not enqueued behind stale work and the stop event is fully drained before
      ``elapsed_time``.

  (d) Distributed: a barrier (all ranks) immediately before AND after each timed
      window aligns ranks so one straggler cannot inflate another rank's window;
      CUDA events are recorded per rank on that rank's device; stdout / returned
      "report" is rank-0 only (every rank still computes its own stats). A
      duck-typed ``dist`` adapter (``.device`` / ``.rank`` / ``.barrier()``)
      decouples this from any specific manager — pass the T0.1
      ``DistributedManager`` instance, or leave it ``None`` and we auto-resolve
      ``torch.distributed`` (if initialised) or fall back to a single-GPU no-op.

This harness is the single trusted reproducer of the design-doc single-device perf
numbers (the docs all report this "drift-cancelled paired-median" event-window
idiom). Validated against ``docs/layernorm_transpose_design.md``: at device-bound
cells ``benchmark_single(iters=1)`` matches the doc harness
(``benchmark/layernorm_transpose/lnt_common.py``'s ``paired_median``) to <1%, and
``benchmark_paired`` reproduces its "median 0.774× vs fold_cp_ops2k" headline (measured
0.779×). The event-window path measures per-call latency (host-submit + device +
launch/event/sync) at ``iters=1`` and amortizes host dispatch toward device time at
larger ``iters``; ``benchmark_single(mode="device")`` reports pure device time via
a self-contained native do_bench-style loop (L2-flushed per-rep CUDA-event timing,
no third-party benchmarking dependency), valid below the per-call host-dispatch
floor (machine-specific; ~5 µs here, ~52 µs on the T0.4 ref) where the event path
bottoms out on host submit, and — unlike the classic do_bench — it also works
distributed (barrier-bracketed reps + ``all_reduce(MAX)`` consensus).

Public API:
  * ``time_callable(fn, iters, device, *, stream=None)`` -> float ms/call.
      The atomic single-window timer (the "fold setup, loop the launch" unit).
  * ``benchmark_single(fn, *, rounds, warmup, iters, device=None, dist=None,
      label=None, print_report=False, mode="event", reduce=None)`` -> ``BenchResult``.
      One target: warmup rounds, then ``rounds`` timed windows; reports the
      median ms/call (+ min/std/raw) and warns on warmup drift. ``mode="event"``
      (default) is the doc-matching event window; ``mode="device"`` is pure
      device time via a native do_bench-style loop (both modes work distributed).
      ``reduce="max"`` (``"event"`` only) ``all_reduce(MAX)``-reduces the per-rank
      median to the slowest PE's — the FIXED-N consensus timer that stays lockstep
      for distributed kernels with in-kernel collectives (see ``mode``/``reduce``).
  * ``benchmark_paired(targets, *, baseline=None, rounds, warmup, iters,
      device=None, dist=None, print_report=False)`` -> ``PairedResult``.
      Many targets timed interleaved per round; reports per-target median ms and
      drift-cancelled median ratio vs ``baseline``.
  * ``resolve_dist(dist=None)`` -> a ``_DistAdapter`` (the duck-typed shim).
"""

from __future__ import annotations

import dataclasses
import os
import statistics
import sys
import time
from typing import Any, Callable, Mapping

import torch

# A zero-argument callable that launches the work to be timed. The caller is
# responsible for having folded ALL one-time setup (input/weight construction,
# JIT warm-compile, .to(device)) into the closure capture, so that calling it is
# nothing but the kernel/module launch (discipline (a)).
BenchFn = Callable[[], Any]


# ─────────────────────────────────────────────────────────────────────────────
# Distributed adapter (discipline (d)) — decoupled from any specific manager.
# ─────────────────────────────────────────────────────────────────────────────
class _DistAdapter:
    """Duck-typed distributed shim: ``.device`` / ``.rank`` / ``.world_size`` / ``.barrier()``.

    Wraps one of three sources, in priority order resolved by :func:`resolve_dist`:

      1. An explicit manager object exposing ``.device`` + ``.rank`` (e.g. the
         T0.1 ``DistributedManager`` Borg singleton). We call its own
         ``.barrier()`` if it has one, else fall back to ``torch.distributed``.
      2. A live ``torch.distributed`` process group (when ``is_initialized()``).
      3. Nothing — a single-process no-op (``rank=0``, ``world_size=1``,
         ``barrier()`` does nothing). This is the standalone single-GPU path, so
         the same timing functions work unmodified off a torchrun launch.
    """

    def __init__(
        self,
        device: torch.device,
        rank: int,
        world_size: int,
        *,
        manager: Any = None,
        use_torch_dist: bool = False,
        local_rank: int = 0,
    ):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self._manager = manager
        self._use_torch_dist = use_torch_dist
        self._local_rank = local_rank

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        """All-rank barrier; no-op when single-process.

        Prefers a manager-supplied ``barrier()`` (it knows its own group /
        device_ids); otherwise drives ``torch.distributed.barrier`` with the
        CUDA ``device_ids`` form (upstream convention — avoids the NCCL
        host-side-barrier warning and pins the barrier to this rank's device).
        """
        if self._manager is not None and hasattr(self._manager, "barrier"):
            self._manager.barrier()
            return
        if self._use_torch_dist and torch.distributed.is_initialized():
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.distributed.barrier(device_ids=[self._local_rank])
            else:
                torch.distributed.barrier()

    def all_reduce_max(self, value: float) -> float:
        """Max of ``value`` across all ranks (identity when single-process).

        Used by ``mode="device"`` so the returned ``median_ms`` is the SLOWEST
        PE's median and is bit-identical on every rank. Drives
        ``torch.distributed.all_reduce`` with ``ReduceOp.MAX`` on the default
        group (the same group the ``torch.distributed`` ``barrier`` path uses);
        NCCL needs a CUDA tensor, gloo a CPU tensor. When no live
        ``torch.distributed`` group is present (e.g. a manager without one) the
        value passes through unchanged.
        """
        if not self.is_distributed:
            return value
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tdev = self.device if self.device.type == "cuda" else torch.device("cpu")
            t = torch.tensor([value], dtype=torch.float64, device=tdev)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            return float(t.item())
        return value

    def all_gather_max_diag(self, value: float) -> tuple[float, int, list[float]]:
        """MAX of ``value`` across ranks, PLUS which rank supplied it and every rank's value.

        Purpose
            Recover the one fact :meth:`all_reduce_max` destroys. ``all_reduce(MAX)`` takes the
            per-rank value as an argument and returns only the maximum, so afterwards nothing can
            distinguish **"every rank was slow"** from **"one rank was slow and MAX reported it"** --
            and those have opposite meanings for a pin. The argmax rank settles it in one integer.

        Why this is not free, and why it is opt-in
            This runs an ``all_gather`` instead of an ``all_reduce`` -- same collective class, but a
            DIFFERENT collective at a different point in a benchmark, so a run using it is not
            timing-comparable with a run that does not. That is why the caller gates it on
            :data:`RANK_DIAG_ENV` and never enables it by default.

        Why the returned max is safe to report anyway
            MAX is exact: it selects an input rather than accumulating, so ``max()`` over the
            gathered ``float64`` values is BIT-IDENTICAL to what ``all_reduce(ReduceOp.MAX)`` would
            have returned for the same inputs. The diagnostic changes what is KNOWN about a number,
            never the number.

        Args:
            value: This rank's value. Must be finite and identical in meaning on every rank --
                a per-rank median in ms here. Every rank MUST call this with the same intent, since
                ``all_gather`` is a collective and one rank skipping it deadlocks the rest.

        Returns:
            ``(max_value, argmax_rank, per_rank_values)``. ``per_rank_values[i]`` is rank ``i``'s
            input, so the tuple is identical on every rank. Ties resolve to the LOWEST rank, which
            is ``list.index``'s behaviour and is stated because a tie is exactly the uninteresting
            case a reader might otherwise over-read.

        Raises:
            RuntimeError: If called with no live ``torch.distributed`` group while ``world_size``
                says otherwise -- gathering is impossible then, and returning a lone value would
                report ``argmax_rank=0`` as if it had been measured.
        """
        if not self.is_distributed:
            return value, self.rank, [value]
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            raise RuntimeError(
                f"all_gather_max_diag needs a live torch.distributed group (world_size="
                f"{self.world_size} but no initialized group). Refusing to report argmax_rank="
                f"{self.rank} from a single unreduced value -- that would look measured and be a "
                f"guess. Unset {RANK_DIAG_ENV} to use the plain all_reduce(MAX) path."
            )
        tdev = self.device if self.device.type == "cuda" else torch.device("cpu")
        src = torch.tensor([value], dtype=torch.float64, device=tdev)
        buf = torch.empty(self.world_size, dtype=torch.float64, device=tdev)
        torch.distributed.all_gather_into_tensor(buf, src)
        per_rank = [float(x) for x in buf.tolist()]
        hi = max(per_rank)
        return hi, per_rank.index(hi), per_rank


def resolve_dist(dist: Any = None, *, device: torch.device | None = None) -> _DistAdapter:
    """Resolve a :class:`_DistAdapter` from an explicit manager, ``torch.distributed``, or nothing.

    :param dist: an already-built ``_DistAdapter`` (returned as-is), OR a manager
        object exposing ``.device`` and ``.rank`` (and optionally ``.barrier`` /
        ``.world_size`` / ``.local_rank``), OR ``None`` to auto-resolve.
    :param device: device override for the no-op / torch.distributed fallback
        (defaults to ``torch.cuda.current_device()`` when CUDA is available).
    :return: an adapter usable by :func:`benchmark_single` / :func:`benchmark_paired`.
    """
    if isinstance(dist, _DistAdapter):
        return dist

    if dist is not None:
        # Manager object (duck-typed). Pull what it exposes; fall back sensibly.
        mgr_device = getattr(dist, "device", None) or device or _default_device()
        rank = int(getattr(dist, "rank", 0))
        world_size = int(getattr(dist, "world_size", 1))
        local_rank = int(
            getattr(
                dist,
                "local_rank",
                rank % max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 0,
            )
        )
        return _DistAdapter(mgr_device, rank, world_size, manager=dist, local_rank=local_rank)

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        dev = device or _default_device()
        local_rank = rank % max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 0
        return _DistAdapter(dev, rank, world_size, use_torch_dist=True, local_rank=local_rank)

    # Single-process no-op.
    return _DistAdapter(device or _default_device(), rank=0, world_size=1)


#: Env var that turns on the per-rank timing diagnostic. Unset/``0`` -> the shipped
#: ``all_reduce(MAX)`` path, byte-for-byte. Set to ``1`` -> :meth:`_DistAdapter.all_gather_max_diag`
#: additionally records WHICH rank set the reported number. Read at CALL time, not import time, so a
#: test (or an operator mid-session) can flip it without re-importing the module.
RANK_DIAG_ENV = "CPO_BENCH_RANK_DIAG"


def rank_diag_enabled() -> bool:
    """Whether the per-rank timing diagnostic is on.

    Returns:
        True when :data:`RANK_DIAG_ENV` is set to something other than ``""``/``"0"``. Any other
        value (``"1"``, ``"true"``, ``"yes"``) enables it -- a diagnostic that silently stayed off
        because the operator wrote ``true`` instead of ``1`` is worse than one that is too eager.
    """
    return os.environ.get(RANK_DIAG_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _consensus_max(da: _DistAdapter, value: float) -> tuple[float, int | None, list[float] | None]:
    """Reduce ``value`` to the slowest rank's, and say WHICH rank that was when asked.

    The single seam between the shipped reduction and the diagnostic one, so the two cannot drift
    and neither call site has to know about the env var.

    Args:
        da: The resolved adapter. When ``world_size == 1`` this is an identity in both modes.
        value: This rank's median ms.

    Returns:
        ``(reduced_value, argmax_rank_or_None, per_rank_or_None)``. **With the diagnostic OFF the
        second and third are always None** -- not a placeholder rank -- so a consumer cannot mistake
        "not measured" for "rank 0". The reduced value is identical in both modes (MAX is exact).
    """
    if not da.is_distributed:
        return value, None, None
    if rank_diag_enabled():
        reduced, argmax, per_rank = da.all_gather_max_diag(value)
        # THE READER. Without this the diagnostic is computed and discarded on every path -- the
        # harness, which is the only one that runs two trees, never mentions these fields. Printed
        # rather than returned-and-hoped-for because the consumers are shell drivers grepping a
        # log, and printed on rank 0 only because 16 identical copies bury the number. Guarded by
        # the same predicate that produced the values, so a run without the flag is byte-identical.
        if da.rank == 0:
            spread = (
                (max(per_rank) / min(per_rank) - 1.0) * 100.0
                if per_rank and min(per_rank)
                else float("nan")
            )
            print(
                f"    [rank-diag] argmax_rank={argmax} reduced_ms={reduced:.6f} "
                f"spread_max_over_min={spread:+.2f}% per_rank_ms={[round(v, 6) for v in per_rank]}",
                flush=True,
            )
        return reduced, argmax, per_rank
    return da.all_reduce_max(value), None, None


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Result records.
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class BenchResult:
    """Aggregate over the timed rounds of a single target (per-round ms/call)."""

    label: str | None
    median_ms: float
    min_ms: float
    std_ms: float
    rounds: int
    iters: int
    raw_ms: list[float]  # one ms/call per timed round
    #: Which rank's median became ``median_ms`` under a MAX reduction, or None when the
    #: diagnostic was off / the run was single-process. None means NOT MEASURED -- never rank 0.
    argmax_rank: int | None = None
    #: Every rank's median ms, index == rank, or None when the diagnostic was off. Present only
    #: under :data:`RANK_DIAG_ENV`; see :meth:`_DistAdapter.all_gather_max_diag` for the cost.
    per_rank_ms: list[float] | None = None


@dataclasses.dataclass
class PairedResult:
    """Per-target paired-median timings vs an optional common baseline.

    ``targets`` maps label -> dict with keys ``median_ms`` (median ms/call over
    rounds), ``median_ratio`` (drift-cancelled median of per-round
    target/baseline, or ``None`` if no baseline), and ``raw_ms`` (per-round
    ms/call). ``baseline_median_ms`` is the baseline's own median ms (or ``None``).
    """

    targets: dict[str, dict[str, Any]]
    baseline_label: str | None
    baseline_median_ms: float | None
    rounds: int
    iters: int


# ─────────────────────────────────────────────────────────────────────────────
# The atomic single-window timer — discipline (a) + (c).
# ─────────────────────────────────────────────────────────────────────────────
def time_callable(
    fn: BenchFn, iters: int, device: torch.device, *, stream: torch.cuda.Stream | None = None
) -> float:
    """ms per call = one CUDA-event window over ``iters`` back-to-back ``fn()`` launches / iters.

    This is the indivisible timing unit. The caller must have folded all one-time
    setup into ``fn`` (discipline (a)) so the window contains only kernel work.
    We ``synchronize`` before recording ``start`` (so it is not enqueued behind
    stale work) and after ``stop`` (so the event is drained before
    ``elapsed_time``) — discipline (c). Events are recorded on ``stream`` (the
    current stream by default); a target that switches the current stream
    internally without restoring it would silently under-report, so we assert it
    did not.

    :param fn: zero-arg launch closure.
    :param iters: back-to-back launches inside the single event window (>= 1).
    :param device: CUDA device the work runs on.
    :param stream: stream to record events on; default ``current_stream(device)``.
    :return: mean ms per single ``fn()`` call.
    """
    if device.type != "cuda":
        raise RuntimeError(f"CUDA-event timing requires a CUDA device; got {device}")
    if iters < 1:
        raise ValueError(f"iters must be >= 1; got {iters}")
    bench_stream = stream if stream is not None else torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record(bench_stream)
    for _ in range(iters):
        fn()
    end.record(bench_stream)
    torch.cuda.synchronize(device)
    cur = torch.cuda.current_stream(device)
    if cur != bench_stream:
        raise RuntimeError(
            f"target switched the current CUDA stream and did not restore it: "
            f"expected {bench_stream}, got {cur} (CUDA events bound to one stream "
            f"do not measure work on another — silent under-reporting)"
        )
    return start.elapsed_time(end) / iters


# ─────────────────────────────────────────────────────────────────────────────
# The additive launch-overhead model — measure the per-window bias, don't pay to hide it.
# ─────────────────────────────────────────────────────────────────────────────
# ── two "launch overhead"s, and they are NOT the same number ──────────────────────────────────
# `measure_launch_overhead` below returns `C`, the per-WINDOW cost of the EVENT-timed path, in ms:
# the host work that happens after a window's opening `synchronize()` and before the device has
# anything to run. `host_dispatch_us` returns the per-CALL host submit cost, in us, from a bare loop
# with no synchronisation inside the span. They answer different questions and are measured by
# different instruments; a caller that reaches for the wrong one gets a plausible number.
#
# Measured, so the gap is not a matter of taste: on the cp8 TriMul workflow `C` solved per cell is
# ~300 us while the same solve against a trivial one-element `add_` reads 3.7 us. That is the
# refutation of "C is a property of the launch path" -- it is a property of the CALLABLE, because
# the Python preamble that runs before the launch belongs to the callable and not to the driver.
# See `docs/fix_test_setup.md` item B4.


#: Loop counts for :func:`host_dispatch_us`. Two, because agreement between them is the ONLY
#: available evidence that the measurement is dispatch and not queue backpressure.
#:
#: Measured on an Intel Xeon Silver 4314 with an H100 SXM5, sweeping n over one entry point at three
#: shapes whose device times span 7.7 us to 499 us:
#:
#:     shape                     device      n=10    n=50   n=200   n=2000
#:     128x256x256 L7             7.7 us     28.6    23.3    23.3     22.1
#:     4096x2048x2048            65.7 us     30.5    24.2    22.7     31.6
#:     1000000x512x128          498.8 us     28.2    21.8    21.2    243.3
#:
#: Both ends are wrong for different reasons. At n=10 the first calls and timer granularity inflate
#: it. At n=2000 a long kernel overruns the CUDA launch queue, the host BLOCKS inside the launch,
#: and the loop silently stops measuring dispatch and starts measuring throttled device time -- the
#: 243 us reading is half the kernel's own 499 us, not a host cost. 50 and 200 agree to ~7% wherever
#: the measurement is valid, and diverge exactly when it is not.
_HOST_N_SMALL, _HOST_N_LARGE = 50, 200

#: How far the two loop counts may disagree before the reading is refused as backpressure-tainted.
#: Set from the ~7% agreement observed across a 65x range of device times, with margin.
_HOST_N_AGREEMENT = 0.25

#: How many (n=50, n=200) PAIRS are measured, each count's readings then medianed before comparing.
#:
#: **One pair was not enough, and the failure it produced was a false one.** Measured on an
#: UNTOUCHED tree, the check refused a perfectly good reading in 4 of 20 runs -- and 6 of 20 on a
#: tree whose dispatch was independently measured to be FASTER, so its fire rate carried no
#: information about the code under it. One observed fire was while timing the REFERENCE, a bare
#: `GemmSm90` launch that no kernel change can reach.
#:
#: **Medianing, not a wider band and not a retry.** Each reading is already a MEAN over its loop, so
#: per-call jitter is averaged out and the thing that trips the check is a rare additive burst -- an
#: OS descheduling event landing inside one loop. Averaging more would only collect more bursts; a
#: median rejects any that land in a minority of the pairs. A wider band would instead have to
#: choose between admitting contaminated readings and refusing good ones, since the two causes
#: differ in KIND rather than degree: backpressure is systematic, a hiccup is not. Retrying until a
#: pair happens to agree was tried and rejected for a subtler reason -- it is three chances to pass,
#: so a genuinely backpressured kernel that agrees once by luck slips through, which a median of
#: robust estimates does not allow.
#:
#: The pairs are INTERLEAVED (small, large, small, large, ...) rather than all-smalls-then-all-larges
#: so that a machine slowing down partway through cannot manufacture a disagreement between the two
#: counts -- the same pairing discipline `bench_timing.benchmark_paired` uses for its ratios.
_HOST_N_SAMPLES = 3


def host_dispatch_us(fn: Callable[[], object], device=None) -> float:
    """Host CPU time per call, in microseconds, for one kernel entry point.

    Purpose
        The second of the two references. A launch-bound cell's ``mode="event"`` time is NOT its
        kernel's time -- measured here, an 8x8-through-a-tile GEMM reads 27 us of event time against
        7 us of device time, because the GPU sits idle inside the event window waiting for Python.
        That floor is host-CPU-bound, so it moves with the machine while a device-side reference
        reports no change at all: across two H100 SXM5 boxes the floor shifted 1.6x while the matmul
        reference read 0.989x. Pinning it separately is what makes the difference visible.

    Semantics
        Times ``fn`` in a bare loop with NO synchronization inside the timed span. The launch is
        async, so what this measures is the host-side submit path -- the repo's per-call Python
        preamble (epilogue args, scheduler args, ``data_ptr()``, the ``jit_cache`` lookup, the
        autotune gate) plus the driver's launch. TVM-FFI removes the tensor marshalling but none of
        that, which is why ~21 us remains on a fast machine.

        **The primary protection is the loop COUNT, not the agreement check.** Backpressure begins
        when the number of pending launches exceeds the CUDA launch queue depth (~1024), which is a
        function of the COUNT and not of how slow the kernel is: at n=2000 a 499 us kernel read
        243 us/call -- half its own device time, dressed up as a plausible dispatch cost -- while at
        n=50 and n=200 the same kernel reads 21.5 us. Both counts here sit an order of magnitude
        below the queue depth, so neither can overrun it.

        The two-count agreement check is therefore a cheap consistency assertion rather than the
        thing that makes this correct: it would catch a driver or device with a much shallower
        queue, and it costs one extra short loop. It is NOT expected to fire in normal use, and it
        does not fire for any kernel currently gated.

        Dispatch is shape-INDEPENDENT for a given entry point (measured: ~7% across a 65x range of
        device times), so callers should measure it once at a TINY shape, where no backpressure is
        possible, rather than per cell.

    Args:
        fn: Zero-argument callable issuing exactly one launch through the entry point being
            characterized. Must already have been called once so JIT compile is not in the loop, and
            must NOT synchronize -- a sync inside makes this measure device time by construction.
        device: Device to drain before and after the timed span. None uses the current device.

    Returns:
        Microseconds of host time per call, from the larger loop count, on the first of
        `_HOST_N_SAMPLES` interleaved pairs.

    Raises:
        RuntimeError: If the two counts' MEDIANS disagree by more than `_HOST_N_AGREEMENT`, naming
            both. Refused rather than averaged together: the two numbers are not two samples of one
            quantity, they are one valid measurement and one contaminated one, and averaging them
            produces a number that is neither. Medianed WITHIN each count rather than judged on a
            single pair because one pair fired on noise in 4 of 20 runs -- see `_HOST_N_SAMPLES`.
    """

    def once(n):
        """Mean host us/call over `n` launches, device drained before and after the span."""
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        elapsed = time.perf_counter() - t0
        torch.cuda.synchronize(device)
        return elapsed / n * 1e6

    pairs = [(once(_HOST_N_SMALL), once(_HOST_N_LARGE)) for _ in range(_HOST_N_SAMPLES)]
    small = statistics.median(p[0] for p in pairs)
    large = statistics.median(p[1] for p in pairs)
    if abs(large - small) / min(small, large) > _HOST_N_AGREEMENT:
        raise RuntimeError(
            f"host dispatch reading is not stable across loop counts: the median of "
            f"{_HOST_N_SAMPLES} samples read {small:.1f} us at "
            f"n={_HOST_N_SMALL} vs {large:.1f} us at n={_HOST_N_LARGE}. The long-loop reading is "
            f"almost certainly queue backpressure -- the kernel is slow enough that the launch "
            f"queue filled and the host blocked in the launch, so the loop measured throttled "
            f"DEVICE time. Measure dispatch at a TINY shape, where the device drains faster than "
            f"the host can submit."
        )
    return large


def _launch_overhead_probe(dev: torch.device) -> BenchFn:
    """Build the cheapest possible zero-arg CUDA launch, for :func:`measure_launch_overhead`.

    Purpose
        The probe's only job is to put SOMETHING on the stream so a window measures a real launch
        path. Its device time CANCELS in the two-point solve, so the requirement is cheapness and
        determinism, not resemblance to any workload.

    Semantics
        A one-element in-place add: one kernel, no allocation per call (the tensor is captured by
        the closure), no host<->device copy, no synchronisation of its own.

    Input requirements
        ``dev`` must be a CUDA device, and must be the current device when the closure is called --
        otherwise the launch lands on another device and the window times an empty stream.

    Returns:
        A zero-arg callable that launches exactly one trivial kernel.
    """
    t = torch.ones(1, device=dev)
    return lambda: t.add_(1.0)


def measure_launch_overhead(
    device: torch.device | None = None,
    dist: Any = None,
    *,
    probe_iters: tuple[int, int] = (1, 20),
    rounds: int = 15,
    warmup: int = 3,
    stream: torch.cuda.Stream | None = None,
) -> float:
    """Measure ``C``, the FIXED per-window cost of the event-timed path, in ms.

    Purpose
        :func:`time_callable` obeys ``measured(N) = device + C/N``, where ``N`` is ``iters``: ONE
        event pair brackets ``N`` back-to-back launches and the elapsed time is divided by ``N``.
        ``C`` is the host cost of getting the first launch onto a queue that the window's opening
        ``synchronize()`` just drained, plus the two event records.

        **``C`` is a property of the CALLABLE, not of the launch path -- this docstring used to
        claim the opposite, and the claim was MEASURED FALSE.** The premise was that host code
        submitting work does not know which kernel it is submitting, so ``C`` could be measured once
        with any cheap kernel and subtracted everywhere. Measured: this function's trivial probe
        reads ``C = 3.7 us`` while the cp8 TriMul workflow's own two-point solve reads ~300 us, an
        80x gap. The Python preamble that runs before the launch -- epilogue args, scheduler args,
        ``data_ptr()``, the ``jit_cache`` lookup, the autotune gate -- belongs to the CALLABLE, and
        it is most of ``C``. So the probe below characterises the FLOOR of the event-window path,
        not any particular workload's ``C``, and it must not be subtracted from another callable's
        measurement.

        What the gate uses instead is :func:`benchmark_extrapolated`, which runs the SAME two-point
        solve against the callable being measured. The alternative both replace is making ``C/N``
        negligible by running ``N = 20`` -- converting an ADDITIVE bias into a MULTIPLICATIVE cost
        of 20x the launches on every cell, forever.

        A GLOBAL probe-and-subtract regime was also built and measured, and is NOT the shipped one:
        with one global ``C = 325.9 us`` the cp8 gate passed 12 of 13, failing the smallest cell at
        1.033x against a 1.025 band, because a single global ``C`` leaves a residual of
        ``(C_cell - C_probe) / device`` -- 42 us on a 1.726 ms pin, which is the whole band. It
        stays reachable behind ``CPO_PERF_PROBE_C`` and would become correct the moment the pins
        were re-harvested under it. See ``docs/fix_test_setup.md`` items B4, G4 and J3.

        Measured on the cp8 TriMul workflow: ``C`` is ~0.3 ms, which is 20% of a 1.76 ms cell and
        0.08% of a 21 ms one. Suppressing it by brute force cost 7740 launches per gate cell
        REGARDLESS of the cell's expense -- 13.6 s of device time for the cheapest cell and 475 s
        for the dearest.

    Semantics
        Times the trivial probe at two ``iters`` values and solves the two-point system exactly::

            m[n1] = device + C/n1
            m[n2] = device + C/n2
            C     = (m[n1] - m[n2]) / (1/n1 - 1/n2)

        The probe's own ``device`` term cancels in the difference, which is why the probe need not
        resemble the workload; cheapness is what CONDITIONS the solve, by making ``C`` a large
        fraction of both measurements. Each ``m[n]`` is a median over ``rounds`` windows, so one slow
        window cannot set ``C``.

        MAX-reduced across ranks like every other number this timer reports: ``C`` is per-rank HOST
        cost and the gate's subject is the slowest PE. The reduce also makes the value rank-uniform,
        which anything subtracted from a collective-bearing measurement must be -- a per-rank
        subtraction would make the reduced median depend on which rank was slowest at probe time.

    Input requirements
        ``probe_iters`` must be two DISTINCT positive ints -- equal values make the solve singular.
        ``rounds >= 1``. ``warmup >= 1`` is strongly advised: the first window after process start
        carries module-load and allocator-warm cost that is NOT ``C``, and at ``warmup=0`` that
        lands in the estimate. ``device`` must be CUDA (or None to resolve from ``dist``), and must
        be current.

    Returns:
        ``C`` in milliseconds, MAX-reduced across ranks. **May be NEGATIVE** on a path where the
        larger-``N`` probe contends with itself; that is a legitimate reading of "no measurable
        per-window cost", it subtracts correctly, and it is NOT clamped -- clamping would hide the
        one signal that says the model does not fit here.

    Raises:
        ValueError: ``probe_iters`` is not two distinct positive ints, or ``rounds < 1``.
        RuntimeError: the resolved device is not CUDA (from :func:`time_callable`).
    """
    n1, n2 = probe_iters
    if n1 == n2 or n1 < 1 or n2 < 1:
        raise ValueError(f"probe_iters must be two distinct positive ints; got {probe_iters!r}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1; got {rounds}")
    da = resolve_dist(dist, device=device)
    dev = device if device is not None else da.device
    probe = _launch_overhead_probe(dev)
    for _ in range(warmup):
        time_callable(probe, max(n1, n2), dev, stream=stream)
    m = {
        n: statistics.median([time_callable(probe, n, dev, stream=stream) for _ in range(rounds)])
        for n in (n1, n2)
    }
    c = (m[n1] - m[n2]) / (1.0 / n1 - 1.0 / n2)
    reduced, _, _ = _consensus_max(da, c)
    return reduced


def benchmark_extrapolated(
    fn: BenchFn,
    *,
    probe_iters: tuple[int, int] = (1, 4),
    rounds: int = 35,
    warmup: int = 5,
    device: torch.device | None = None,
    dist: Any = None,
    label: str | None = None,
    stream: torch.cuda.Stream | None = None,
    reduce: str | None = "max",
) -> BenchResult:
    """Time ``fn`` at TWO small ``iters`` values and report the ``iters -> inf`` limit.

    Purpose
        The event window obeys ``measured(N) = device + C/N``. ``C`` is the host cost of dispatching
        ``fn()`` once: :func:`time_callable` records the start event BEFORE the Python call runs, so
        the device idles for exactly one dispatch before the first kernel arrives, and that gap is
        then divided by ``N``.

        The shipped gate suppresses ``C`` by running ``N = 20``, which converts an ADDITIVE bias into
        a MULTIPLICATIVE cost -- 20x the launches on every cell, forever, regardless of how expensive
        the cell is. Measured on the cp8 TriMul workflow: 7740 launches per gate cell, 13.6 s of
        device time for the cheapest cell and 475 s for the dearest.

        Solving for ``C`` instead removes the bias for the price of a second cheap point.

    Semantics
        Measures a median-over-``rounds`` at each of ``probe_iters`` and solves the two-point system
        exactly::

            C      = (m[n1] - m[n2]) / (1/n1 - 1/n2)
            device = m[n2] - C/n2

        **``C`` is per-callable and must be solved HERE, not supplied from a probe.** Measured: a
        trivial ``t.add_(1.0)`` gives ``C = 3.7 us`` while the TriMul workflow gives ~300 us, because
        ``C`` is the Python cost of ``fn`` itself (a DTensor round trip), not of the CUDA launch path.
        A generic probe would under-correct by two orders of magnitude. See
        :func:`measure_launch_overhead` for the probe that established this.

        Validated against the shipped pins: extrapolating the cp8 workflow cells from the (1, 20)
        pair reproduced all six to within **0.58%**, against a 2.5% band.

    Input requirements
        ``probe_iters`` must be two DISTINCT positive ints -- equal values make the solve singular.
        The pair trades launches against noise: the solve weights the two medians by
        ``n2/(n2-n1)`` and ``n1/(n2-n1)``, so a close pair amplifies their noise. ``(1, 2)`` costs 3
        windows-worth of launches at ~2.24x amplification, ``(1, 4)`` costs 5 at ~1.37x, ``(1, 20)``
        costs 21 at ~1.0x. Default ``(1, 4)``: 4x fewer launches than ``iters=20`` at 1.37x noise,
        i.e. ~0.82% against a 2.5% band.

        ``fn`` must be a zero-arg closure with all setup folded in (bench discipline (a)), and its
        host cost must be STABLE across calls -- a callable that memoises on first use reports a
        larger ``C`` than it will ever pay again, and the extrapolation then over-corrects. The
        ``warmup`` windows exist to settle exactly that.

    Returns:
        A :class:`BenchResult` whose ``median_ms`` is the extrapolated ``device`` term.
        ``raw_ms`` holds the LARGER probe's per-round values (uncorrected) for diagnosis, and
        ``iters`` reports ``n2``. ``min_ms``/``std_ms`` likewise describe that probe, NOT the
        extrapolate -- an extrapolated value has no per-round samples of its own, and inventing them
        would misrepresent a derived number as a measured one.

    Raises:
        ValueError: ``probe_iters`` not two distinct positive ints, or ``reduce`` invalid.
        RuntimeError: the resolved device is not CUDA (from :func:`time_callable`).
    """
    n1, n2 = probe_iters
    if n1 == n2 or n1 < 1 or n2 < 1:
        raise ValueError(f"probe_iters must be two distinct positive ints; got {probe_iters!r}")
    if reduce not in (None, "max"):
        raise ValueError(f"reduce must be None or 'max'; got {reduce!r}")
    da = resolve_dist(dist, device=device)
    dev = device if device is not None else da.device
    for _ in range(warmup):
        if da.is_distributed:
            da.barrier()
        time_callable(fn, n2, dev, stream=stream)
    per_probe: dict[int, list[float]] = {n1: [], n2: []}
    for _ in range(rounds):
        # Both probes inside ONE round, back to back between the same barrier pair, so a drift that
        # moves one moves the other -- the solve is a DIFFERENCE, so a drift landing between two
        # separately-timed loops would be amplified by n2/(n2-n1) instead of cancelling.
        if da.is_distributed:
            da.barrier()
        per_probe[n1].append(time_callable(fn, n1, dev, stream=stream))
        per_probe[n2].append(time_callable(fn, n2, dev, stream=stream))
        if da.is_distributed:
            da.barrier()
    m1 = statistics.median(per_probe[n1])
    m2 = statistics.median(per_probe[n2])
    c = (m1 - m2) / (1.0 / n1 - 1.0 / n2)
    median_ms = m2 - c / n2
    argmax_rank: int | None = None
    per_rank_ms: list[float] | None = None
    if reduce == "max":
        median_ms, argmax_rank, per_rank_ms = _consensus_max(da, median_ms)
    raw = per_probe[n2]
    return BenchResult(
        label=label,
        median_ms=median_ms,
        min_ms=min(raw),
        std_ms=statistics.pstdev(raw) if len(raw) > 1 else 0.0,
        rounds=rounds,
        iters=n2,
        raw_ms=raw,
        argmax_rank=argmax_rank,
        per_rank_ms=per_rank_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-target benchmark — warmup + timed rounds, with the drift guard.
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_single(
    fn: BenchFn,
    *,
    rounds: int = 35,
    warmup: int = 5,
    iters: int = 20,
    device: torch.device | None = None,
    dist: Any = None,
    label: str | None = None,
    stream: torch.cuda.Stream | None = None,
    print_report: bool = False,
    mode: str = "event",
    reduce: str | None = None,
    subtract_overhead: float | None = None,
) -> BenchResult:
    """Time one target: ``warmup`` untimed windows, then ``rounds`` timed windows.

    Reports the MEDIAN ms/call over the timed rounds (robust to the occasional
    slow window) plus min/std/raw. In a distributed adapter the timed region is
    barrier-bracketed per round (discipline (d)) and only rank 0 prints; every
    rank still returns its own ``BenchResult``.

    **Two timing modes** (``mode=``):

      * ``"event"`` (default) — the doc-matching event-WINDOW path described
        above: each timed window records one ``torch.cuda.Event`` pair around
        ``iters`` back-to-back ``fn()`` launches / ``iters``. With ``iters=1``
        this is the design docs' per-call latency (host-submit + device-kernel +
        launch/event/sync) and reproduces the ``layernorm_transpose`` doc method
        to <1% at device-bound cells; with larger ``iters`` it amortizes
        per-call host dispatch toward device time. This is the FIXED-N path
        (``iters`` is fixed + identical on every rank), so — unlike ``"device"`` —
        it is SAFE for distributed kernels with in-kernel collectives; opt into
        cross-rank consensus with ``reduce="max"`` (below).
      * ``"device"`` — pure DEVICE time via a self-contained native do_bench-style
        loop (:func:`_do_bench_native`): an L2-flushed per-rep CUDA-event clearing
        loop that removes per-call host dispatch and L2-cache reuse, so it works
        BELOW the per-call host-dispatch floor (machine-specific; ~5 µs here, ~52 µs
        on the T0.4 ref) where the event-window path bottoms out on host submit.
        ``rounds``/``warmup``/``iters`` are reinterpreted as wall-clock ms budgets
        (see below); ``raw_ms`` carries the per-replica samples. Use this to report
        kernel-only time; keep ``"event"`` to report what a real per-call site pays.
        BOTH modes now support distributed timing (barrier-bracketed, per-rank);
        ``"device"`` additionally reduces the per-rank median via ``all_reduce(MAX)``
        so ``median_ms`` is the slowest PE's and identical on every rank.

        **Distributed caveat — ``"device"`` uses ADAPTIVE rep-counting.**
        :func:`_do_bench_native` derives its per-rank ``n_rep`` from that rank's
        measured ``est_ms`` (``n_rep = rep_ms / est_ms``), so ranks whose kernels
        run at different speeds execute DIFFERENT iteration counts. That is fine
        for a per-rank-independent / collective-FREE kernel, but it DESYNCS a
        kernel with in-kernel collectives (nvshmem puts, device barriers): a
        faster rank issues more puts than a slower one and the ranks deadlock.
        For such kernels use ``mode="event", reduce="max"`` — its fixed, identical
        ``iters`` keeps every rank in lockstep while still yielding the slowest-PE
        consensus. (There is no static guard: an in-kernel collective is not
        detectable from ``fn``; this is a usage contract.)

    :param fn: zero-arg launch closure (setup already folded in — discipline (a)).
    :param rounds: timed windows (``"event"``); in ``"device"`` mode it sets the
        repetition budget ``rep_ms = rounds * iters * a_floor`` so a longer
        ``rounds``/``iters`` buys more samples (clamped ≥ 20 ms).
    :param warmup: untimed warm windows run first (``"event"`` — discipline (c));
        in ``"device"`` mode it is the warmup ms budget (clamped ≥ 25 ms).
    :param iters: back-to-back launches per window (``"event"``); in ``"device"``
        mode it only scales the rep budget (the loop manages its own repetition).
    :param device: CUDA device; defaults to the resolved adapter's device.
    :param dist: distributed adapter / manager / ``None`` (see :func:`resolve_dist`).
        Honored by BOTH modes (``"device"`` adds an ``all_reduce(MAX)`` consensus).
    :param label: free-form label for the report.
    :param stream: optional stream for the event window (``"event"`` only).
    :param print_report: print a one-block human summary on rank 0.
    :param mode: ``"event"`` (default, doc-matching window) or ``"device"`` (native
        pure-device loop).
    :param reduce: cross-rank reduction of the returned ``median_ms`` (``"event"``
        mode only, opt-in). ``None`` (default) → per-rank median (byte-identical to
        the pre-existing behavior). ``"max"`` → ``all_reduce(MAX)`` the per-rank
        median so ``median_ms`` is the SLOWEST PE's and bit-identical on every rank
        (``raw_ms``/``min_ms``/``std_ms`` stay per-rank diagnostics); single-process
        → identity no-op. This is the FIXED-N barrier'd consensus timer — the
        counterpart to ``"device"``'s built-in MAX, and what a distributed autotuner
        of in-kernel-collective kernels needs. In ``mode="device"`` the per-rank
        median is ALWAYS MAX-reduced (that mode's defining consensus), so ``reduce``
        governs only ``"event"``; ``reduce="max"`` is accepted with ``"device"`` as a
        redundant restatement and ``reduce=None`` does NOT disable the device reduce.
        Any other value raises ``ValueError``.
    :param subtract_overhead: ``C`` from :func:`measure_launch_overhead`, in ms, or None.
        When given, ``C / iters`` is subtracted from EVERY window before the median is taken, so the
        reported number is the ``device`` term of ``measured(N) = device + C/N`` rather than the
        measurement at this particular ``iters``. That is what lets a caller run ``iters=1`` -- the
        20x this timer's callers pay exists only to shrink ``C/N``, and a term that is measured does
        not need to be shrunk.

        Subtracted PER WINDOW rather than from the final median because the two differ whenever the
        median is not linear in its inputs, and because ``raw_ms``/``min_ms``/``std_ms`` should be
        overhead-free too -- a consumer reading ``min_ms`` off a corrected median would otherwise get
        an uncorrected number from the same object.

        ``None`` (default) subtracts nothing and is byte-identical to the previous behaviour, so no
        existing caller changes until it opts in. Applies to ``mode="event"`` only; ``mode="device"``
        runs its own loop with a different window structure and REFUSES a non-None value rather than
        silently ignoring it.
    :return: a :class:`BenchResult`.
    """
    if subtract_overhead is not None and mode != "event":
        raise ValueError(
            f"subtract_overhead applies to mode='event' only (the C/N model describes ITS window "
            f"structure); got mode={mode!r}. Pass None, or use mode='event'."
        )
    if reduce not in (None, "max"):
        raise ValueError(f"reduce must be None or 'max'; got {reduce!r}")
    da = resolve_dist(dist, device=device)
    dev = device if device is not None else da.device

    if mode == "device":
        return _benchmark_single_device(
            fn,
            rounds=rounds,
            warmup=warmup,
            iters=iters,
            dev=dev,
            da=da,
            label=label,
            print_report=print_report,
        )
    if mode != "event":
        raise ValueError(f"mode must be 'event' or 'device'; got {mode!r}")

    # The per-window bias, already divided by `iters`: one window's fixed cost is amortised over the
    # `iters` launches inside it, exactly as `time_callable` divides the elapsed time by `iters`.
    bias = 0.0 if subtract_overhead is None else subtract_overhead / iters

    # Warmup (untimed) — burn JIT caches + settle clocks (discipline (c)).
    for _ in range(warmup):
        if da.is_distributed:
            da.barrier()
        time_callable(fn, iters, dev, stream=stream)

    # Corrected by the SAME bias as the timed windows: `_maybe_warn_warmup_drift` compares the
    # warmup tail against the bench head, so leaving one side uncorrected would shift that ratio by
    # `bias / ms` and move the 2x tripwire for no physical reason.
    warmup_ms = (
        [time_callable(fn, iters, dev, stream=stream) - bias for _ in range(min(warmup, 3))]
        if warmup
        else []
    )

    raw_ms: list[float] = []
    for _ in range(rounds):
        # Barrier BEFORE and AFTER the timed window so a straggler rank can't
        # bleed into this rank's measurement (discipline (d); no-op single-proc).
        if da.is_distributed:
            da.barrier()
        ms = time_callable(fn, iters, dev, stream=stream) - bias
        if da.is_distributed:
            da.barrier()
        raw_ms.append(ms)

    median_ms = statistics.median(raw_ms)
    # Distributed consensus (opt-in via reduce="max"): reduce this rank's median to the SLOWEST PE's
    # so median_ms is bit-identical on every rank — the fixed-N barrier'd counterpart to mode="device"'s
    # built-in MAX. The reduce is a collective every rank reaches (reduce= is passed identically on all
    # ranks, and it runs after the per-round barriers), so no asymmetry/deadlock; raw/min/std stay
    # per-rank. Single-proc → all_reduce_max is an identity no-op (gated on is_distributed).
    argmax_rank: int | None = None
    per_rank_ms: list[float] | None = None
    if reduce == "max":
        median_ms, argmax_rank, per_rank_ms = _consensus_max(da, median_ms)

    res = BenchResult(
        label=label,
        median_ms=median_ms,
        min_ms=min(raw_ms),
        std_ms=statistics.pstdev(raw_ms) if len(raw_ms) > 1 else 0.0,
        rounds=rounds,
        iters=iters,
        raw_ms=raw_ms,
        argmax_rank=argmax_rank,
        per_rank_ms=per_rank_ms,
    )
    _emit_rank_diag(res, da)
    if warmup_ms:
        _maybe_warn_warmup_drift(warmup_ms, raw_ms, da)
    if print_report and da.is_rank0:
        _print_single(res, dev, da)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Pure-device-time mode — native do_bench-style loop (host dispatch removed).
# Self-contained: torch + CUDA events only, no third-party benchmarking dep.
# ─────────────────────────────────────────────────────────────────────────────
def _l2_flush_buffer(dev: torch.device) -> torch.Tensor:
    """An int8 clearing buffer >= the L2 cache size, zeroed before each timed rep.

    Matches the canonical do_bench L2-eviction: a fixed 256 MiB buffer, widened to
    the device's actual ``L2_cache_size`` if that ever exceeds 256 MiB, so a
    memory-bound kernel is not inflated by input data lingering in L2 across
    replicas. ``.zero_()`` writes every byte, evicting prior tenants.
    """
    l2_bytes = 256 * 1024 * 1024
    try:
        l2_bytes = max(l2_bytes, int(torch.cuda.get_device_properties(dev).L2_cache_size))
    except Exception:  # noqa: BLE001 — property may be absent on some builds
        pass
    return torch.empty(l2_bytes, dtype=torch.int8, device=dev)


def _do_bench_native(
    fn: BenchFn, *, warmup_ms: float, rep_ms: float, dev: torch.device, da: _DistAdapter
) -> list[float]:
    """Native pure-device timing loop — the do_bench algorithm, torch + CUDA events only.

    Reproduces the load-bearing pieces of the canonical ``do_bench``:

      1. **Estimate → budgets:** one warm ``fn()`` + sync, then a 5-call flushed
         estimate → ``est_ms``; ``n_warmup = max(1, warmup_ms/est_ms)`` and
         ``n_rep = max(1, rep_ms/est_ms)``.
      2. **L2 flush:** ``cache.zero_()`` (see :func:`_l2_flush_buffer`) BEFORE each
         timed rep so memory-bound kernels aren't inflated by L2 reuse.
      3. **Per-rep event pairs:** ``n_rep`` ``(start,end)`` pairs; per rep flush,
         ``start.record()``, ``fn()``, ``end.record()``; ONE ``synchronize`` after
         the loop; ``times[i] = start[i].elapsed_time(end[i])``.

    Distributed (what do_bench cannot): ``da.barrier()`` immediately before AND
    after the timed loop so a straggler rank can't bleed into another rank's
    window; each rank times its own reps. Returns the per-rep ms samples (the
    caller reduces the median across ranks).
    """
    with torch.cuda.device(dev):
        # Warm once + settle (do_bench: fn(); synchronize()).
        fn()
        torch.cuda.synchronize(dev)
        cache = _l2_flush_buffer(dev)

        # Estimate est_ms/call over 5 flushed calls (do_bench step 1).
        est_start = torch.cuda.Event(enable_timing=True)
        est_end = torch.cuda.Event(enable_timing=True)
        est_start.record()
        for _ in range(5):
            cache.zero_()
            fn()
        est_end.record()
        torch.cuda.synchronize(dev)
        est_ms = est_start.elapsed_time(est_end) / 5.0
        if est_ms <= 0.0:
            est_ms = 1e-3  # coarse timer / trivially-fast kernel — avoid div-by-zero

        n_warmup = max(1, int(warmup_ms / est_ms))
        n_rep = max(1, int(rep_ms / est_ms))
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]

        # Warm-up (untimed).
        for _ in range(n_warmup):
            fn()

        # Barrier BEFORE the timed loop so all ranks start aligned (no-op single-proc).
        if da.is_distributed:
            da.barrier()
        for i in range(n_rep):
            cache.zero_()  # flush L2 before each timed rep
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize(dev)
        # Barrier AFTER so a straggler can't bleed into a peer's next window.
        if da.is_distributed:
            da.barrier()

        return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _benchmark_single_device(
    fn: BenchFn,
    *,
    rounds: int,
    warmup: int,
    iters: int,
    dev: torch.device,
    da: _DistAdapter,
    label: str | None,
    print_report: bool,
) -> BenchResult:
    """``benchmark_single(mode="device")``: pure DEVICE time via a native do_bench loop.

    :func:`_do_bench_native` runs ``fn`` inside an L2-flushed per-rep CUDA-event
    loop, so the reported time is the pure DEVICE kernel time with per-call
    Python/host-dispatch removed — meaningful BELOW the per-call host-dispatch
    floor (machine-specific; ~5 µs here, ~52 µs on the T0.4 ref) where the
    event-window path bottoms out on host submit. Fully self-contained (torch +
    CUDA events); no third-party benchmarking dependency.

    ``warmup``/``rounds``/``iters`` (which are *counts* in the event path) are mapped
    onto wall-clock *ms budgets*: ``warmup_ms = max(warmup * 5, 25)`` and
    ``rep_ms = max(rounds * iters * 0.05, 20)`` — a longer event-mode warmup/rounds
    therefore buys a longer settle/sample budget too, with the canonical floors
    (25 ms warm / 20 ms rep) as the minimum. ``raw_ms`` holds the per-replica
    samples; ``median_ms`` is their median.

    **Distributed** (previously unsupported): each rank times its own reps behind a
    before/after barrier; the per-rank median is reduced via ``all_reduce(MAX)`` so
    the returned ``median_ms`` is the SLOWEST PE's and IDENTICAL on every rank
    (``raw_ms``/``min_ms``/``std_ms`` stay per-rank diagnostics).

    **Distributed caveat — ADAPTIVE rep-counting is NOT collective-safe.** The rep
    count is derived per-rank from that rank's measured ``est_ms``
    (``n_rep = rep_ms / est_ms`` in :func:`_do_bench_native`), so ranks running at
    different speeds execute DIFFERENT iteration counts. This is safe only for
    per-rank-INDEPENDENT / collective-FREE kernels; a kernel with in-kernel
    collectives (nvshmem puts, device barriers) DESYNCS — the faster rank issues
    more puts than the slower one and they deadlock. For collective-in-kernel
    workloads use ``benchmark_single(mode="event", reduce="max")`` instead: its
    fixed, identical ``iters`` keeps every rank in lockstep and still returns the
    slowest-PE consensus. (No code guard — an in-kernel collective is not
    statically detectable from ``fn``; this is a documented usage contract.)
    """
    if dev.type != "cuda":
        raise RuntimeError(f"device-time mode requires a CUDA device; got {dev}")

    warmup_ms = max(float(warmup) * 5.0, 25.0)
    rep_ms = max(float(rounds) * float(iters) * 0.05, 20.0)
    raw_ms = _do_bench_native(fn, warmup_ms=warmup_ms, rep_ms=rep_ms, dev=dev, da=da)
    if not raw_ms:
        raw_ms = [float("nan")]

    local_median = statistics.median(raw_ms)
    # Distributed consensus: report the slowest PE's median, identical on all ranks. Under
    # RANK_DIAG_ENV this additionally records WHICH rank that was -- see _consensus_max.
    median_ms, argmax_rank, per_rank_ms = _consensus_max(da, local_median)

    res = BenchResult(
        label=label,
        median_ms=median_ms,
        min_ms=min(raw_ms),
        std_ms=statistics.pstdev(raw_ms) if len(raw_ms) > 1 else 0.0,
        rounds=len(raw_ms),
        iters=1,
        raw_ms=raw_ms,
        argmax_rank=argmax_rank,
        per_rank_ms=per_rank_ms,
    )
    _emit_rank_diag(res, da)
    if print_report and da.is_rank0:
        _print_single(res, dev, da)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Multi-target paired-median benchmark — discipline (b).
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_paired(
    targets: Mapping[str, BenchFn],
    *,
    baseline: BenchFn | None = None,
    baseline_label: str = "baseline",
    rounds: int = 35,
    warmup: int = 5,
    iters: int = 20,
    device: torch.device | None = None,
    dist: Any = None,
    stream: torch.cuda.Stream | None = None,
    print_report: bool = False,
) -> PairedResult:
    """Time several targets INTERLEAVED per round; report drift-cancelled median ratios.

    Each round (discipline (b)): time ``baseline`` (if given) then every target
    back-to-back via :func:`time_callable`. After all rounds, per target report
    ``median_ms`` (median over rounds) and ``median_ratio`` = median over rounds
    of ``target_ms / baseline_ms``. Because a common-mode clock drift in a round
    scales the baseline and the target by the same factor, it cancels in the
    per-round ratio — the median ratio is drift-robust even when the raw ms
    wander. Mirrors the ``bench_trimul.py`` core, generalised + reusable.

    Each timed window is barrier-bracketed when distributed (discipline (d)); a
    target that raises is dropped (recorded with ``status``) so one bad config
    does not abort the sweep.

    :param targets: label -> zero-arg launch closure (setup folded in).
    :param baseline: optional zero-arg closure timed first each round for ratios.
    :param baseline_label: label recorded for the baseline.
    :param rounds: timed rounds.
    :param warmup: untimed warm rounds (full back-to-back window).
    :param iters: back-to-back launches per window.
    :param device / dist / stream: as in :func:`benchmark_single`.
    :param print_report: print a per-target table on rank 0.
    :return: a :class:`PairedResult`.
    """
    da = resolve_dist(dist, device=device)
    dev = device if device is not None else da.device
    labels = list(targets.keys())
    dropped: dict[str, str] = {}

    def _window(fn: BenchFn) -> float:
        if da.is_distributed:
            da.barrier()
        ms = time_callable(fn, iters, dev, stream=stream)
        if da.is_distributed:
            da.barrier()
        return ms

    # Warmup the whole back-to-back window (baseline + every target).
    for _ in range(warmup):
        if baseline is not None:
            _window(baseline)
        for lb in list(labels):
            try:
                _window(targets[lb])
            except Exception as e:  # noqa: BLE001 — surface as a dropped target, keep sweeping
                dropped[lb] = f"{type(e).__name__}: {e}"
                labels.remove(lb)

    # Timed rounds: baseline first, then each surviving target, back-to-back.
    base_ms_rounds: list[float] = []
    tgt_ms_rounds: dict[str, list[float]] = {lb: [] for lb in labels}
    tgt_ratio_rounds: dict[str, list[float]] = {lb: [] for lb in labels}
    for _ in range(rounds):
        base_ms = _window(baseline) if baseline is not None else float("nan")
        base_ms_rounds.append(base_ms)
        for lb in list(labels):
            try:
                ms = _window(targets[lb])
            except Exception as e:  # noqa: BLE001
                dropped[lb] = f"{type(e).__name__}: {e}"
                labels.remove(lb)
                tgt_ms_rounds.pop(lb, None)
                tgt_ratio_rounds.pop(lb, None)
                continue
            tgt_ms_rounds[lb].append(ms)
            if (
                baseline is not None and base_ms == base_ms and base_ms > 0
            ):  # base_ms==base_ms drops NaN
                tgt_ratio_rounds[lb].append(ms / base_ms)

    med = statistics.median
    ok_base = [m for m in base_ms_rounds if m == m]
    base_med = med(ok_base) if (baseline is not None and ok_base) else None

    out: dict[str, dict[str, Any]] = {}
    for lb in labels:
        if not tgt_ms_rounds.get(lb):
            out[lb] = {"status": "no timed rounds"}
            continue
        out[lb] = {
            "median_ms": med(tgt_ms_rounds[lb]),
            "median_ratio": (med(tgt_ratio_rounds[lb]) if tgt_ratio_rounds[lb] else None),
            "raw_ms": tgt_ms_rounds[lb],
            "status": "ok",
        }
    for lb, reason in dropped.items():
        out.setdefault(lb, {"status": reason})

    res = PairedResult(
        targets=out,
        baseline_label=baseline_label if baseline is not None else None,
        baseline_median_ms=base_med,
        rounds=rounds,
        iters=iters,
    )
    if print_report and da.is_rank0:
        _print_paired(res, dev, da)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Reporting / drift guard (rank-0 gated).
# ─────────────────────────────────────────────────────────────────────────────
def _maybe_warn_warmup_drift(
    warmup_ms: list[float], bench_ms: list[float], da: _DistAdapter
) -> None:
    """Warn (rank-0, stderr) if the warmup tail and bench head differ by > 2x.

    Two-sided: fires when warmup is much slower (insufficient JIT amortisation)
    or much faster (suspicious cache-cold bench). Same heuristic as the upstream
    harness.
    """
    if len(warmup_ms) < 1 or len(bench_ms) < 3:
        return
    warmup_tail = float(statistics.mean(warmup_ms[-3:]))
    bench_head = float(statistics.mean(bench_ms[:3]))
    if bench_head <= 0.0 or warmup_tail <= 0.0:
        return
    ratio = warmup_tail / bench_head
    if (ratio > 2.0 or ratio < 0.5) and da.is_rank0:
        print(
            f"warning: warmup may be insufficient — warmup-tail mean = {warmup_tail:.4f} ms, "
            f"bench-head mean = {bench_head:.4f} ms (ratio {ratio:.2f})",
            file=sys.stderr,
        )


def _dist_tag(da: _DistAdapter) -> str:
    return f" [rank {da.rank}/{da.world_size}]" if da.is_distributed else ""


def _emit_rank_diag(res: BenchResult, da: _DistAdapter) -> None:
    """Print the argmax rank when the diagnostic ran -- INDEPENDENTLY of ``print_report``.

    Purpose
        Give the diagnostic an exit that does not depend on the consumer. It previously lived inside
        :func:`_print_single`, which runs only under ``print_report=True``; **every real consumer
        leaves that default False**, so on the production path the argmax was computed and then had
        nowhere to go. Measured: a five-launch venue A loop with the env correctly set produced
        zero argmax lines, because the field was discarded by the caller (`.median_ms`) and the
        banner was behind a flag nobody passes.

    Why the guard is the FIELD and not an env read
        ``res.argmax_rank is not None`` is true only when :func:`_consensus_max` actually gathered,
        so the OFF path is byte-identical -- no print, no env lookup, and this sits after the timed
        window in any case. Re-reading the env here would be a second source of truth for one fact.

    Why rank 0 only
        ``all_gather`` gives every rank the same per-rank list, so one line carries the whole
        result; 16 identical copies would bury the numbers the caller prints.

    Args:
        res: The freshly built result. ``argmax_rank``/``per_rank_ms`` are either both set (the
            diagnostic ran) or both None (it did not); a half-populated pair cannot occur because
            :func:`_consensus_max` returns them together.
        da: The adapter, for ``is_rank0``. A single-process run never reaches the body, since
            ``_consensus_max`` short-circuits and leaves ``argmax_rank`` None.

    Returns:
        None. Emits on stdout, matching the channel the perf gates already print through -- and
        note that channel is subject to pytest's capture, so a measuring invocation needs ``-s``.
        The per-rank predicate line the gate logs is the canary for that: if IT is missing too,
        capture ate the run rather than the diagnostic failing.
    """
    if res.argmax_rank is None or not da.is_rank0:
        return
    per = res.per_rank_ms or []
    spread = max(per) / min(per) if per and min(per) > 0 else float("nan")
    # `sample_ms=`, NOT `median=`. The value IS a median (this call's, MAX-reduced across ranks),
    # but `median=` ALREADY denotes a different quantity in the same log: `calibration.assert_cell`
    # prints `median=... rel_std=` for the HARVEST median over 15 such calls. Two emitters, one
    # token, different referents -- and this line fires per call, so it is always the FIRST match.
    # An excursion detector built on `grep -m1 "median="` therefore read one sample as the harvest
    # median and came out ANTI-CORRELATED with the excursions it was flagging: 3 false positives,
    # 2 real excursions missed. Renaming here removes the ambiguity at the SOURCE, which is
    # strictly better than requiring every downstream extractor to anchor around it.
    # DO NOT "unify" this back to `median=` for consistency; the inconsistency is the point.
    print(
        f"    [rank-diag] argmax_rank={res.argmax_rank} "
        f"slow/fast={spread:.3f}x sample_ms={res.median_ms:.5f}\n"
        f"    [rank-diag] per_rank_ms={['%.5f' % v for v in per]}\n"
        f"    [rank-diag] NOTE: {RANK_DIAG_ENV} is set -- this run used an all_gather instead of "
        f"the shipped all_reduce(MAX), so its timings are NOT comparable with a pin.",
        flush=True,
    )


def _print_single(res: BenchResult, dev: torch.device, da: _DistAdapter) -> None:
    name = torch.cuda.get_device_name(dev.index if dev.index is not None else 0)
    hdr = f"=== bench {res.label} ===" if res.label else "=== bench ==="
    print(
        f"{hdr}{_dist_tag(da)}\n"
        f"device : {dev} ({name})\n"
        f"rounds : {res.rounds} (warmup folded) x {res.iters} iters/window\n"
        f"median (ms/call): {res.median_ms:.4f}\n"
        f"min    (ms/call): {res.min_ms:.4f}\n"
        f"std    (ms/call): {res.std_ms:.4f}"
    )


def _print_paired(res: PairedResult, dev: torch.device, da: _DistAdapter) -> None:
    name = torch.cuda.get_device_name(dev.index if dev.index is not None else 0)
    print(f"=== paired bench ==={_dist_tag(da)}")
    print(f"device : {dev} ({name})  rounds={res.rounds} x {res.iters} iters/window")
    if res.baseline_median_ms is not None:
        print(f"{res.baseline_label:<28} {res.baseline_median_ms:>10.4f} ms (baseline)")
    for lb, e in sorted(res.targets.items(), key=lambda kv: kv[1].get("median_ms", float("inf"))):
        if e.get("status") != "ok":
            print(f"{lb:<28} {'N/A':>10}  ({e.get('status')})")
            continue
        r = e["median_ratio"]
        rs = f"{r:.3f}x" if r is not None else "  --  "
        print(f"{lb:<28} {e['median_ms']:>10.4f} ms   ratio={rs}")


# ─────────────────────────────────────────────────────────────────────────────
# Cold-compile comparison — sampling discipline for a NON-device measurement.
# ─────────────────────────────────────────────────────────────────────────────
#: The floor on samples per side. Five, not one, and not a taste: a single cold-compile sample
#: produced a FALSE REGRESSION twice in one day of bring-back work, both times pointing the same
#: way. At ``N=256 D=128 cont_pipe`` one sample read ours 9.037 s against main 6.074 s -- +2.96 s,
#: +49%, "we regressed" -- while five samples per side gave medians 6.364 and 6.292, a +1.1%
#: parity. The same thing happened earlier on the LayerNorm subset, where one sample read 3x.
#: Under this repo's bring-back rule a cold-compile regression is a BLOCKER, so the one-sample
#: answer does not merely mislead: it stops a good commit and sends someone hunting a bug that is
#: not there.
MIN_COLD_COMPILE_SAMPLES = 5


@dataclasses.dataclass(frozen=True)
class ColdCompileComparison:
    """Two sides' cold-compile samples, their medians, and whether the delta is above the noise.

    Attributes:
        baseline: Label of the reference side, e.g. ``"main"``.
        candidate: Label of the side under test, e.g. ``"ours"``. The sign convention follows from
            these two names and nothing else: ``delta_s`` is candidate MINUS baseline, so a
            positive delta means the candidate is SLOWER.
        baseline_samples: The baseline's durations in seconds, in COLLECTION order (not sorted), so
            a reader can see a drift for themselves rather than take the median on trust.
        candidate_samples: The candidate's durations, likewise.
        order: The labels in the exact order ``measure`` was called, for the same reason. Its length
            is ``2 * samples``.
    """

    baseline: str
    candidate: str
    baseline_samples: tuple[float, ...]
    candidate_samples: tuple[float, ...]
    order: tuple[str, ...]

    @property
    def samples(self) -> int:
        """How many samples were taken per side."""
        return len(self.baseline_samples)

    @property
    def baseline_median_s(self) -> float:
        """The baseline's median duration in seconds. MEDIAN, never mean -- a cold compile's
        distribution has a long right tail (a page-cache miss, a scheduler preemption, another job
        landing on the box), and one such sample moves a mean of five by a fifth of its excess
        while moving the median not at all."""
        return statistics.median(self.baseline_samples)

    @property
    def candidate_median_s(self) -> float:
        """The candidate's median duration in seconds."""
        return statistics.median(self.candidate_samples)

    @property
    def delta_s(self) -> float:
        """``candidate_median_s - baseline_median_s``. Positive means the candidate is slower."""
        return self.candidate_median_s - self.baseline_median_s

    @property
    def ratio(self) -> float:
        """``candidate_median_s / baseline_median_s``. 1.0 is parity; above 1.0 is a slowdown."""
        return self.candidate_median_s / self.baseline_median_s

    @property
    def baseline_spread_s(self) -> float:
        """The baseline's observed range, ``max - min``, in seconds."""
        return max(self.baseline_samples) - min(self.baseline_samples)

    @property
    def candidate_spread_s(self) -> float:
        """The candidate's observed range, ``max - min``, in seconds."""
        return max(self.candidate_samples) - min(self.candidate_samples)

    @staticmethod
    def _mad_s(samples) -> float:
        """Median absolute deviation from the median, in seconds.

        Purpose
            A dispersion estimate a SINGLE outlier cannot dominate, reported beside the range
            rather than replacing it. `baseline_spread_s` is ``max - min``: one slow cold compile
            out of five -- a node warm-up, a filesystem stall, an unrelated JIT miss -- moves it by
            the whole excursion, while the MAD moves by roughly nothing.

        Why BOTH are reported and neither is the verdict
            They answer different questions and each is blind where the other sees. The range
            catches a bimodal or long-tailed side that a MAD calls tight; the MAD catches a
            genuinely tight side that one excursion makes the range call noisy. Measured on this
            project's runtime comparator the same day: a cell read MAD **0.97%** -- inside a 5% bar
            -- while its ratio of max to min was **12.06**. A rule on either scalar alone passes
            one of those two cases silently.

        Args:
            samples: The per-sample durations in seconds, in collection order. Order is not read
                here, but the caller must keep it: re-scoring a completed run against a different
                statistic needs the raw series, and a summary cannot be un-summarized.

        Returns:
            The MAD in seconds. ``0.0`` for a single sample -- which is honest (one point exhibits
            no dispersion) and is why it must never be read as "this side is stable".
        """
        vals = list(samples)
        if not vals:
            return 0.0
        med = statistics.median(vals)
        return statistics.median([abs(v - med) for v in vals])

    @property
    def baseline_mad_s(self) -> float:
        """The baseline's median absolute deviation, in seconds. See `_mad_s`."""
        return self._mad_s(self.baseline_samples)

    @property
    def candidate_mad_s(self) -> float:
        """The candidate's median absolute deviation, in seconds. See `_mad_s`."""
        return self._mad_s(self.candidate_samples)

    @property
    def resolvable(self) -> bool:
        """Whether ``|delta_s|`` exceeds the WIDER side's own observed range.

        The honest summary of a cold-compile comparison, and the fact a single sample cannot
        supply. False means the two sides are not distinguishable at this sample count -- report
        parity, or take more samples; it does NOT mean they are equal. True means the shift is
        larger than the noise the run itself exhibited, which is when a delta is worth acting on.

        This is deliberately a PROPERTY and not an exception: whether an unresolvable delta blocks a
        commit is the caller's policy, and a timing primitive that made that call would be deciding
        it for every caller.
        """
        return abs(self.delta_s) > max(self.baseline_spread_s, self.candidate_spread_s)

    def describe(self) -> str:
        """A multi-line report: both medians, both spreads, the delta and whether it resolves.

        Returns:
            Text intended to be pasted into a report verbatim. Both sides' numbers are shown
            because a single ratio hides which side moved -- and a reader who cannot see the spread
            cannot tell a real 1.1% from a noise-dominated 49%.
        """
        return (
            f"cold compile, {self.samples} samples/side, alternated\n"
            f"  {self.baseline:<12} median {self.baseline_median_s:8.3f} s  "
            f"spread {self.baseline_spread_s:7.3f} s  mad {self.baseline_mad_s:7.3f} s  samples "
            f"{[round(v, 3) for v in self.baseline_samples]}\n"
            f"  {self.candidate:<12} median {self.candidate_median_s:8.3f} s  "
            f"spread {self.candidate_spread_s:7.3f} s  mad {self.candidate_mad_s:7.3f} s  samples "
            f"{[round(v, 3) for v in self.candidate_samples]}\n"
            f"  delta {self.delta_s:+.3f} s ({self.ratio:.3f}x, "
            f"{100 * (self.ratio - 1):+.1f}%)  "
            f"resolvable={self.resolvable}\n"
            # `resolvable` is gated on the RANGE alone. Both dispersions are printed above so a
            # reader can see when the two disagree -- a wide range beside a small mad is one
            # excursion, not noise, and is exactly the case where a False here is not "equal".
            f"  (resolvable is gated on the RANGE; compare it against mad before reading a False "
            f"as parity)"
        )

    def __str__(self) -> str:
        """One line: the two medians, the delta and whether it resolves."""
        return (
            f"{self.baseline} {self.baseline_median_s:.3f}s vs {self.candidate} "
            f"{self.candidate_median_s:.3f}s -> {self.delta_s:+.3f}s ({self.ratio:.3f}x, "
            f"resolvable={self.resolvable})"
        )


def compare_cold_compile(
    measure: Callable[[str], float],
    *,
    baseline: str = "main",
    candidate: str = "ours",
    samples: int = MIN_COLD_COMPILE_SAMPLES,
) -> ColdCompileComparison:
    """Compare two sides' COLD compile time with enough samples that the answer means something.

    Purpose
        A cold compile is not a device measurement, so none of the machinery above applies to it --
        no CUDA events, no L2 flush, no warmup (a warmed compile is not a cold one). What it shares
        with a device measurement is the part people get wrong: it is a random variable, and this
        repo treats a cold-compile regression as a BLOCKER. One sample against one sample has twice
        produced a blocker that was not there; see :data:`MIN_COLD_COMPILE_SAMPLES` for both
        measurements.

    Semantics
        Three disciplines, all enforced here rather than left to the caller:

        1. **At least five samples per side.** Fewer is refused, not silently accepted.
        2. **The median**, never the mean -- see :attr:`ColdCompileComparison.baseline_median_s`.
        3. **The two sides ALTERNATE, in ABBA order.** Machine load drifts over the minutes a
           sweep takes; running side A's five samples and then side B's five hands the whole drift
           to one side. Plain ABAB alternation fixes most of that but still gives one side the
           same position in every pair, and a compile is sensitive to position: the first process
           after a gap pays page-cache and import costs the second does not. ABBA rotates that
           slot. The realized order is returned in ``order`` so a reader can check it.

        **This function does NOT spawn processes, and that is the design decision.** A genuine cold
        compile needs a FRESH interpreter -- ``@jit_cache`` memoizes in-process, so a second call in
        the same process measures a dict lookup -- but what "fresh" means belongs to the caller:
        which interpreter, which environment (``CPO_CACHE_ENABLED=0``), which ``sys.path`` (a
        bring-back comparison puts the two trees on different ones), which config to compile, and
        how to get the duration back. A primitive that took a command template would be a
        subprocess runner with a median inside it, and its own test would have to spawn processes
        -- slow, flaky, and unrunnable on a box without a GPU. Consuming a ``measure`` callable
        keeps this pure: its test drives it with the literal numbers from a real measurement, and
        the subprocess mechanics stay in the one place that knows them.

    Args:
        measure: ``measure(label) -> seconds``. Called ``2 * samples`` times, alternating; each
            call MUST perform one genuine cold compile of the side named by ``label``, in a fresh
            process, and return its wall duration in seconds. Returning a cached or in-process
            duration is not detectable here beyond the positivity check below -- the guarantee this
            function makes is about the SAMPLING, not about the freshness.

            A caller who already holds two lists of durations can adapt them
            (``lambda side: next(iters[side])``), but the alternation guarantee is then void: the
            order was fixed before this function saw it, and ``order`` will describe a sequence
            that did not happen.
        baseline: Label of the reference side. Passed verbatim to ``measure``.
        candidate: Label of the side under test. Must differ from ``baseline``; two sides with one
            name would make the returned samples unattributable.
        samples: Samples PER SIDE. At least :data:`MIN_COLD_COMPILE_SAMPLES`.

    Returns:
        A :class:`ColdCompileComparison`.

    Raises:
        ValueError: If ``samples`` is below the floor; if ``baseline == candidate``; or if
            ``measure`` returns a duration that is not finite and positive -- a zero or negative
            "compile" means the measurement did not measure a compile (a warm cache, a mis-parsed
            log line), and a median would launder that into a plausible number.
    """
    if samples < MIN_COLD_COMPILE_SAMPLES:
        raise ValueError(
            f"samples={samples} is below the floor of {MIN_COLD_COMPILE_SAMPLES} per side. A cold "
            f"compile is a random variable and this repo treats a regression in it as a blocker: "
            f"one sample per side has twice reported a regression that five samples showed to be "
            f"parity (+49% -> +1.1% at one shape, 3x -> parity at another). Take five, or do not "
            f"claim a delta."
        )
    if baseline == candidate:
        raise ValueError(
            f"baseline and candidate are both {baseline!r}. The labels are what `measure` is asked "
            f"for and what the samples are attributed to, so two sides cannot share one name."
        )

    order: list[str] = []
    got: dict[str, list[float]] = {baseline: [], candidate: []}
    for i in range(samples):
        # ABBA: the side that goes first alternates, so neither side always occupies the
        # first-after-a-gap slot, which is the one that pays the page-cache and import costs.
        pair = (baseline, candidate) if i % 2 == 0 else (candidate, baseline)
        for label in pair:
            value = float(measure(label))
            if not (value > 0.0 and value < float("inf")):
                raise ValueError(
                    f"measure({label!r}) returned {value!r} on sample {len(got[label]) + 1}. A cold "
                    f"compile takes a finite positive time; a zero, a negative or an infinity means "
                    f"the measurement did not measure a compile -- a warm cache, a mis-parsed log "
                    f"line, or a process that failed before compiling. Refused here rather than "
                    f"folded into a median that would look plausible."
                )
            order.append(label)
            got[label].append(value)

    return ColdCompileComparison(
        baseline=baseline,
        candidate=candidate,
        baseline_samples=tuple(got[baseline]),
        candidate_samples=tuple(got[candidate]),
        order=tuple(order),
    )
