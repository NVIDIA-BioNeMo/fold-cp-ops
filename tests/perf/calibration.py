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

"""Pin a perf cell against its OWN measured spread, on a clock-locked machine.

**Lock the clock before running these gates**::

    sudo nvidia-smi -lgc 1290        # see PIN_CLOCK_MHZ

Without it the gates can fail on cells that have not changed. That is a deliberate trade, arrived
at after trying the alternative, and the reasoning is worth keeping because it is not obvious.

**What this replaces.** Bands used to be hand-set constants (``_TOL = 1.10`` / ``_TOL_LAUNCH =
1.25``) chosen from eyeballed spreads -- too tight for the launch-bound cells, too loose for the
compute-bound ones, and unrelated to what any cell actually did. Each cell now carries its own
measured ``rel_std`` from harvest time, and :func:`band_for` derives the band from it. Nobody picks
a number, and a cell whose noise changes gets a band that changes with it.

**What this DELIBERATELY does not do: correct for the machine's speed.** An earlier version divided
every measurement by ``reference_now / reference_pinned``, taken from one ``torch.matmul``. It was
tried thoroughly and removed, because a single reference can only cancel a slowdown that hits it and
the target equally, and on real hardware almost nothing does:

* **compute vs bandwidth.** Unlocked, the matmul reference sped up 22% while bandwidth-bound
  LayerNorm sped up 6.6% -- HBM clock does not follow SM clock. Dividing credited LayerNorm with a
  speed-up it never got: 23 false failures in one run, none of them a real regression.
* **host vs device.** A launch-bound cell's time is host dispatch, which a device-side reference
  cannot see at all. Across two H100 SXM5 boxes the dispatch floor moved 1.6x while the reference
  read 0.989x.
* **power regime.** A 700 W-capped heavy kernel sustains ~1335 MHz while the short bursty reference
  holds ~1425 -- so even two compute-bound workloads do not share a clock.

Each fix for one of those was a new reference class, and the next mismatch was always one more
kernel away. Meanwhile the correction added the reference's own noise to every band, and cost three
separate rounds of false failures to discover. Locking the clock removes the whole class outright:
locked, cells reproduce to **0.26-0.7% across processes**, which no correction ever achieved.

**So the clock is a precondition, not something the arithmetic pretends to absorb.** When a gate
fails, :func:`clock_advice` checks the current SM clock against `PIN_CLOCK_MHZ` and says which of
the two explanations applies, because a ratio alone cannot tell "the kernel changed" from "the box
is in a different state".

**What is pinned, and what a test run does.**

    pin time    N_SAMPLES_PIN_TIME samples of the cell -> median_ms, rel_std   (its own noise)
    test time   N_SAMPLES_TEST_TIME samples, median    -> compared against median_ms * band_for(cell)

One residual, measured and unsolved: some heavy compute-bound cells are BISTABLE unlocked (the
``(1e6, 768, 384)`` GEMM reads ~1.09 or ~1.46 ms, 2-of-4 each way) and no amount of warm-up
converges them -- only the lock does. That is the sharpest reason the lock is a precondition rather
than a recommendation.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from typing import Callable, Optional, Sequence

# Module level, not deferred: `pins` imports nothing from this module (its only mention of
# calibration is prose), so there is no cycle, and `assert_cell`'s default argument needs the
# constant at DEFINITION time.
import tests.perf.pins as pins_mod

# `host_dispatch_us` MOVED to the package (`fold_cp_ops/_internal/bench_timing.py`) and is imported
# back under its original name so every existing `calibration.host_dispatch_us` call site is
# unchanged. It moved because it is a TIMING PRIMITIVE, and CLAUDE.md's benchmarking rule says a
# benchmark times through `bench_utils` -- but `bench_utils` re-exports the package module, and a
# `benchmark/**` file importing from `tests/` is refused by
# `scripts/guard_no_test_import_in_perf_bench.py`. So while it lived here it was reachable from the
# gates and from nothing else, and a benchmark author following the rule could not find it. That is
# not hypothetical: this function was re-invented twice during the perf-gate rebuild -- once as a
# CUDA-event two-point solve that shipped a 6x-wrong `C` into the gate, once as a `perf_counter`
# loop with a per-call `synchronize` that read 3x high -- both times because writing a new
# instrument was faster than finding this one. See `docs/fix_test_setup.md` item J.
from fold_cp_ops._internal.bench_timing import host_dispatch_us  # noqa: F401

#: Clocks are left FREE-RUNNING for both harvesting and testing. Deliberately, and symmetrically.
#:
#: Locking is available (``sudo nvidia-smi -lgc``) and does make any single measurement far more
#: reproducible -- locked at 1290 MHz the noisiest cell repeats to 0.26% across processes against
#: ~35% free-running. It is still the wrong tool here, because **locking shifts the mean**: an H100's
#: free-running clock is set by the 700 W power cap and therefore by the workload, so a locked pin
#: and a free-running measurement are samples of two DIFFERENT distributions. No amount of sampling
#: reconciles that, and the mismatch is not small -- 1290-locked pins compared against free-running
#: runs produced 23 failures in one suite, none of them a real regression.
#:
#: What makes free-running workable is sampling, on both sides equally. The medians converge as
#: 1/sqrt(N): the worst cell's across-process spread is 35% at one sample and 5.1% at 25. With
#: `N_SAMPLES_PIN_TIME` and `N_SAMPLES_TEST_TIME` it sits under the ~10% the bands allow.
#:
#: The symmetry is the load-bearing part. Harvest and test must use the SAME clock policy and the
#: same kind of estimator, or the gate compares two distributions and calls the difference a
#: regression.
_CLOCKS_ARE_FREE_RUNNING = True

#: How many independent medians to take when PINNING, for the reference AND for each cell. The
#: standard deviation of a sample of 15 is itself uncertain by roughly ``1/sqrt(2*14)`` ~ 19%, which
#: `_STD_INFLATE` covers; going much higher costs harvest time for a second-order gain.
N_SAMPLES_PIN_TIME = 15

#: How many medians a TEST run takes of the CELL itself, before taking their median.
#:
#: **Not for averaging down noise -- for rejecting a transient.** One `measure()` is already a
#: median over `rounds` timed rounds inside ``bench_utils.benchmark_single``, and consecutive
#: `measure()` calls agree to about 0.6% (25 back-to-back samples of the 4096x2048x1024 epilogue
#: cell spanned 0.03682-0.03726). So a second sample buys almost nothing against ordinary spread.
#: What it buys is immunity to an excursion that spans a WHOLE `benchmark_single` call, which the
#: inner median cannot see and which no amount of `rounds` would fix: the same cell has been
#: observed at 0.0399 in a full session, ~8% off, against that 0.6% spread. A median of 3 discards
#: one such sample outright; a single sample cannot tell it from a regression.
#:
#: 3 rather than more because the cost is linear in the gate's whole runtime and the failure mode
#: being defended against is ONE bad sample. Two would not have a majority to fall back on.
#:
#: The band is deliberately NOT narrowed by this count, though taking a median of 3 does shrink the
#: measurement's own error. Tightening here would be reasoning from the wrong number: the pinned
#: ``rel_std`` is harvested WITHIN one process, where most cells bottom out at the `_MIN_REL_STD`
#: floor, so it already understates the across-session variation the gate actually faces. Until the
#: pins are harvested across processes, the extra margin is doing real work.
N_SAMPLES_TEST_TIME = 9

#: Cumulative sample counts the pinned path ESCALATES through, cheapest first.
#:
#: **Why a ladder and not a smaller constant.** A gate cell costs `samples x 43 windows x 20 iters`
#: launches -- 7740 at 9 samples -- and that count is FIXED regardless of how expensive the cell is,
#: so the workflow's 61 ms cell spends 475 s of pure device time to re-confirm a number that one
#: sample already reproduces to ~0.6% against a 2.5% band. Simply cutting the constant to 1 was
#: rejected: the comment above says the outer samples are partly absorbing across-session variation
#: that the within-process `rel_std` cannot see, and a marginal regression is far likelier to slip
#: through a 1-sample median than a 9-sample one.
#:
#: The ladder keeps both. A cell COMFORTABLY inside its band (see `_EARLY_EXIT_FRAC`) stops at one
#: sample; a cell anywhere near the band escalates and is judged on the median of all 9 -- the
#: SAME estimator, and therefore the same verdict, as before. Cost moves to where the doubt is.
#: Measured on the 12 cp8 cells: every ratio landed in 0.998-1.008 against a 1.025 band, i.e. all
#: of them well inside the early-exit threshold, so the suite pays 1 sample per cell and 9 only
#: when a cell is actually in question.
_ESCALATION = (1, 3, 9)

#: Fraction of the band's width within which ONE sample is allowed to settle the cell.
#: At 0.5 a cell must measure at least twice as close to the pin as the band allows before the
#: cheap path is trusted; anything beyond that buys the full sample count. Not tunable for speed --
#: raising it trades the gate's power against marginal regressions for wall time.
_EARLY_EXIT_FRAC = 0.5

#: Multiplier on the pinned relative spread that defines the band. 4 sigma on a roughly-normal
#: statistic is a ~1-in-16000 false-failure rate per cell; across ~30 cells run a few times a day
#: that is about one spurious red per two years, which is the rate at which people still read a
#: failure instead of re-running it.
SIGMA = 4.0

#: Inflation on the pinned spread, covering the uncertainty of estimating a standard deviation from
#: `N_SAMPLES_PIN_TIME` observations. Not a fudge factor: at n=15 the sample std is itself uncertain by
#: ~19%, so a band built from it without this is systematically a little too tight.
_STD_INFLATE = 1.25

#: Floor on the relative spread, in case a cell measures suspiciously stable at pin time. A band of
#: literally zero width would fail on the first bit of jitter; 0.5% is below every spread measured
#: on this box and keeps a degenerate harvest from producing an unusable gate.
_MIN_REL_STD = 0.005

#: Per-measurement jitter that does NOT scale with the cell, in milliseconds. Added to the pinned
#: spread in quadrature by :func:`band_for`, which widens short cells and leaves long ones alone.
#:
#: **This is the across-process variation the pinned `rel_std` cannot see, measured.** `rel_std` is
#: harvested WITHIN one process (see `N_SAMPLES_PIN_TIME`), so it captures how much a cell moves
#: between back-to-back samples and nothing about how much it moves between SESSIONS -- which is
#: the only regime the gate is ever evaluated in. The comment on `N_SAMPLES_TEST_TIME` has said so
#: for as long as this file has existed; what was missing was the number.
#:
#: Measured on one H100 SXM5 node: 4 whole-suite harvests of the same tree, same node, fresh
#: autotune cache each time, so the only difference between rounds is the process.
#:
#:     median_ms   across    intra   excess   abs_us   cell
#:       0.01247   0.0134   0.0051   0.0124    0.155   ('shape', 4096, 2048, 64, 1)
#:       0.01563   0.0106   0.0050   0.0094    0.147   ('terms', False, True)
#:
#: `across` is the spread of that cell's median over four whole-suite runs; `intra` is what the pin
#: file records. The excess is `sqrt(across^2 - intra^2)` and, converted to absolute time, lands on
#: the same ~0.15 us for cells of different duration -- which is what makes it a FIXED jitter rather
#: than a proportional one, and therefore an absolute term rather than a bigger _MIN_REL_STD.
#: Only cells that failed in EVERY run are used: a cell prints its median just when it fails, so any
#: cell that passed somewhere has a censored sample and a spread estimate biased low.
#:
#: Read as a fixed cost it is a few hundred nanoseconds of launch-path jitter: irrelevant to a
#: millisecond cell (at 1 ms it contributes 0.015% and vanishes in quadrature) and dominant
#: for a ten-microsecond one, which is exactly the population that was flapping. Four whole-suite
#: runs of two trees produced 28 distinct failing cells of which only 3 failed every time, and the
#: same tree run twice gave 7 failures then 20.
#:
#: Sized from the measurement, NOT from what makes today's failures pass -- a band fitted to an
#: outcome is a band that will be re-fitted at the next outcome.
_ABS_JITTER_MS = 0.000150

#: Relative spread a pin carries across SESSIONS AND MACHINES, for gates whose pin and gate are not
#: taken on the same box. Zero by default: it must be opted into by a gate that has MEASURED it.
#:
#: **Why a second, RELATIVE term when `_ABS_JITTER_MS` already exists.** That one is a fixed
#: few-hundred-nanosecond launch jitter -- dominant on a 10 us cell, nothing on a 250 ms one. The
#: effect here is the opposite shape: a whole-machine difference (clock/power state under a 700 W
#: cap, thermals, a NIC handler that fell back, a node freshly returned from maintenance) scales
#: WITH the cell, so it is proportional and survives on the largest cells. Neither term substitutes
#: for the other, and both add in quadrature because they are independent.
#:
#: **Measured, not fitted to a failure.** 85 A2A-workflow cells pinned on
#: three H100 SXM5 nodes across four allocations and re-measured on a fourth in fresh
#: processes, 2026-08-24. The spread of new-vs-old median, split by the re-measured round count:
#:
#:     rounds   n    sd      max
#:          8  18  1.60%   +4.38%
#:         12  14  1.65%   +5.63%
#:         18  13  1.26%   +3.51%
#:         25  13  1.03%   +1.75%
#:         35  27  0.95%   +2.09%
#:
#: It falls monotonically with rounds, so part of it IS the old 5-round median's own error -- and it
#: does NOT fall to zero. 1.0% is the best-converged subset's sd, and it is an UPPER bound on the
#: pure venue term because even that subset moved rounds. Sized there rather than at the 1.65% that
#: would have cleared every observed failure: a band fitted to an outcome gets re-fitted at the next
#: one. Re-measure when a same-node, same-rounds pairing exists; only 1 of the 85 had one.
#:
#: At the `_MIN_REL_STD` floor this takes the band from 1.025x to 1.056x. That floor is where the
#: observed false failures were: two cells at rel_std=0.005 measured 1.036x and 1.026x against
#: bands of 1.025x and 1.026x -- i.e. the cells with the TIGHTEST possible band are the ones with no
#: room for a machine change, which is the signature of a missing term rather than of a regression.
_CROSS_SESSION_REL = 0.010


@dataclasses.dataclass(frozen=True)
class Stats:
    """A pinned measurement: where it sat, and how much it moved.

    Attributes:
        median_ms: Median of `N_SAMPLES_PIN_TIME` independent medians, in milliseconds.
        rel_std: Standard deviation of those samples, divided by the median. Dimensionless, so a
            cell can be re-pinned on a faster machine without the band changing meaning.
    """

    median_ms: float
    rel_std: float

    def __post_init__(self):
        """Refuse a nonsensical pin at construction rather than at comparison time.

        Raises:
            ValueError: If the median is not positive or the spread is negative. Both mean the
                harvest failed, and a gate built on them would compare against nothing.
        """
        if not self.median_ms > 0:
            raise ValueError(f"median_ms must be positive; got {self.median_ms}")
        if self.rel_std < 0:
            raise ValueError(f"rel_std must be non-negative; got {self.rel_std}")


def summarize(samples: Sequence[float]) -> Stats:
    """Reduce repeated measurements of one thing to the pair a gate needs.

    Args:
        samples: Independent medians, in milliseconds. At least two -- a single sample carries no
            information about spread, and pretending otherwise is how a too-tight band gets pinned.

    Returns:
        A :class:`Stats`. The relative spread is floored at `_MIN_REL_STD`.

    Raises:
        ValueError: If fewer than two samples are given.
    """
    if len(samples) < 2:
        raise ValueError(
            f"need at least 2 samples to estimate a spread; got {len(samples)}. A pin without a "
            f"spread forces a hand-picked band, which is what this module exists to remove."
        )
    med = statistics.median(samples)
    return Stats(median_ms=med, rel_std=max(statistics.stdev(samples) / med, _MIN_REL_STD))


def band_for(
    cell: Stats, *, abs_jitter_ms: float = _ABS_JITTER_MS, cross_session_rel: float = 0.0
) -> float:
    """The multiplier a measurement may exceed the pin by before it counts as a regression.

    Purpose
        Replaces a hand-picked constant with the cell's own measured spread, so each cell is gated
        at its own noise level. A launch-bound cell and a compute-bound one no longer need the same
        band, and nobody has to decide which cells are which.

    Semantics
        ``1 + SIGMA * inflated cell spread``, where the spread combines the cell's pinned
        (within-process) ``rel_std`` with :data:`_ABS_JITTER_MS` in quadrature.

        **The absolute term exists because `rel_std` measures the wrong thing for a short cell.**
        It is harvested inside one process, so it describes back-to-back agreement; the gate runs in
        a fresh process every time, and a fixed few-hundred-nanosecond jitter in the launch path is
        a large FRACTION of a 10 us cell and nothing at all on a 1 ms one. Adding it in quadrature
        rather than thresholding on duration keeps the band continuous -- a cliff at some chosen
        microsecond count would put two neighbouring cells under different rules for no physical
        reason -- and leaves compute-bound cells numerically untouched.

        **There used to be a second term, for a drift correction that no longer exists.** Cells were
        divided by ``reference_now / reference_pinned`` from a single `torch.matmul`, which added
        that reference's own noise to every band and -- worse -- was simply wrong whenever the
        reference and the target were limited by different things. Unlocked, it produced 23 false
        failures in one run: the compute-bound reference sped up 22% while bandwidth-bound LayerNorm
        sped up 6.6%, so dividing credited LayerNorm with a speed-up it never got. The same
        mismatch appeared across host-vs-device and across power regimes.

        Correcting for the clock was always the harder half of the problem and never the reliable
        one. LOCKING it is: locked, cells reproduce to 0.26-0.7% across processes, which no amount
        of correction achieved. So the clock is a precondition, stated in `clock_advice`, rather
        than something the arithmetic pretends to absorb.

    Args:
        cell: The cell's pinned statistics.
        abs_jitter_ms: The fixed, non-scaling part of the noise, in the SAME UNITS as
            ``cell.median_ms``. Defaults to :data:`_ABS_JITTER_MS`, which is milliseconds -- so a
            caller whose "median" is not a duration must pass ``0.0``. `assert_host_dispatch` is
            exactly that caller: its pin is a dimensionless RATIO of two host costs, and dividing a
            millisecond constant by 1.447 gives a number with no meaning. It comes out negligible
            rather than wrong-by-a-lot, which is why it has to be explicit: a silently-negligible
            unit error is one nobody finds.

        cross_session_rel: Relative spread the cell carries between a pin taken on one machine and
            a gate run on another, dimensionless. **Defaults to 0.0, so every existing gate is
            numerically unchanged**; a gate opts in only after measuring it on its own venue --
            see :data:`_CROSS_SESSION_REL`. Passing a value measured elsewhere is how a band gets
            loosened without evidence.

    Returns:
        A multiplier strictly greater than 1.
    """
    # Quadrature, not a sum: the two are independent noise sources, so their variances add and
    # their standard deviations do not. Summing would roughly double the widening on the cells this
    # is aimed at, which would be over-correcting by exactly the amount that hides a real
    # regression.
    effective = math.hypot(cell.rel_std, abs_jitter_ms / cell.median_ms, cross_session_rel)
    return 1.0 + SIGMA * effective * _STD_INFLATE


def clock_advice() -> str:
    """A line for a failure message giving the SM clock the measurement was taken at.

    Purpose
        A perf failure has two very different causes -- the kernel changed, or the box was in an
        unusual state -- and a ratio alone cannot tell them apart. Since nothing here corrects for
        machine state (see `_CLOCKS_ARE_FREE_RUNNING`), the clock is reported as CONTEXT for whoever
        reads the failure.

    Semantics
        Reports, it does not prescribe. In particular it does NOT tell the reader to lock the clock:
        the pins were harvested free-running, so a locked run would be comparing against a different
        distribution and is more likely to cause a failure than to explain one.

    Returns:
        One line naming the current SM clock, or a note if it could not be read.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        now = int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return "  (SM clock unavailable)"
    return (
        f"  SM clock at read time: {now} MHz. Pins are harvested free-running with the same sample "
        f"counts, so clock state should already be absorbed -- do NOT lock the clock to 'fix' this, "
        f"which would compare against a different distribution."
    )


#: The dispatch pins are DIMENSIONLESS RATIOS to a bare `GemmSm90` launch, so there is no pinned
#: host reference constant and no host drift factor.
#:
#: There used to be both. The gate pinned absolute microseconds and divided by a reference
#: measured once per session, and it failed 2 runs in 4 with the drift factor reading 0.86-0.95
#: -- always fast, never slow. The bias was not in the measurement but in the PIN: the reference
#: was harvested by a session fixture at session start while the dispatch pins were harvested
#: minutes later, so the two halves of the implied ratio came from different moments (pinned
#: 44.44/19.25 = 2.31 against a properly paired 45.13/17.3 = 2.61). Re-measuring the reference
#: adjacent at TEST time made it worse, because the pin carried the mismatch either way.
#:
#: Pinning the ratio removes the whole class of error: both halves are measured back-to-back by
#: construction, so there is no moment at which only one was measured, and the CPU cancels
#: without anything having to model it. See `assert_host_dispatch`.


def make_host_reference(device="cuda") -> Callable[[], object]:
    """A zero-argument callable issuing ONE bare ``GemmSm90`` launch at a tiny shape.

    Purpose
        Reports how fast this machine dispatches a CuTe-DSL kernel, so a dispatch pin can be
        compared across machines. The device reference cannot stand in for this: across two H100
        SXM5 boxes the dispatch floor moved 1.6x while the matmul reference read 0.989x.

    Semantics
        **A CuTe-DSL GEMM rather than a torch op, so reference and target share a dispatch CLASS.**
        This is the same rule the device reference follows (see the module docstring): a reference
        is only valid where it is limited by the same thing as the target. Every gated entry submits
        through the CuTe-DSL / TVM-FFI path, and a ``torch.add`` would instead track PyTorch's own
        dispatcher -- a different code path whose cost can move independently across machines and
        torch versions, which would show up as drift the gates would silently absorb.

        **The BARE functor, not a public entry -- and that part is load-bearing.** What gets gated
        is the public entries (`gemm`, `dual_gated_gemm`, `layernorm_fwd`), whose extra cost over
        this reference IS the repo's per-call Python: epilogue args, scheduler args, ``data_ptr()``,
        the ``jit_cache`` lookup, the autotune gate. If the reference went through one of those same
        entries, then ``corrected = now / (now / pin) == pin`` identically -- the reference and the
        target would be independent samples of ONE quantity, so they track each other on any host,
        including when the submit path genuinely regresses. The gate would still pass. Keeping the
        reference one level below everything it calibrates is what leaves a regression somewhere to
        show up.

        Shape is tiny by design: dispatch is shape-independent for a given entry point (measured
        within ~7% across a 65x range of device times), so a tiny shape measures the same quantity
        for the least time and cannot hit launch-queue backpressure.

    Args:
        device: Where to allocate. Must be CUDA; a host reference for a GPU submit path is
            meaningless otherwise.

    Returns:
        The callable. Call it once before timing so the JIT compile lands outside the measured loop.
    """
    import torch

    from tests.kernels.test_gemm_sm90 import _build_operands, run_base_gemm

    m, n, k, ell = HOST_REFERENCE_SHAPE
    a, b, d = _build_operands(m, n, k, ell, torch.bfloat16, "k", "k")
    if str(device) not in ("cuda", str(a.device)):  # operands are built on the current device
        a, b, d = a.to(device), b.to(device), d.to(device)
    return lambda: run_base_gemm(a, b, d, 64, 64, (1, 1), False, True)


#: The tiny shape the host reference dispatches. Small enough that device time (~7 us) cannot
#: approach dispatch (~20 us), and K=8 at bf16 is exactly the 16-byte alignment floor the repo
#: allows -- the smallest legal operand, so nothing about it is arbitrary.
HOST_REFERENCE_SHAPE = (8, 8, 8, 1)


_HOST_REFERENCE_CACHE: dict = {}


def _host_reference(device=None) -> Callable[[], object]:
    """The host-reference callable for `device`, built once and reused.

    Purpose
        :func:`assert_host_dispatch` re-measures the reference next to every target, so building it
        each time would put a JIT lookup and three allocations on a path that runs per gated entry.
        Caching makes the paired measurement cheap enough to be unconditional.

    Args:
        device: The CUDA device, or None for the current one. Cached per device, since a callable
            closed over one device's tensors would silently measure a cross-device launch on another.

    Returns:
        The zero-argument callable from :func:`make_host_reference`, already called once so its JIT
        compile is not inside anyone's timed loop.
    """
    key = str(device) if device is not None else "cuda"
    if key not in _HOST_REFERENCE_CACHE:
        import torch

        ref = make_host_reference(key)
        ref()
        torch.cuda.synchronize(device)
        _HOST_REFERENCE_CACHE[key] = ref
    return _HOST_REFERENCE_CACHE[key]


def repeat_median(measure: Callable[[], float], samples: int) -> list:
    """Call a measurement `samples` times and return the results, for :func:`summarize`.

    Args:
        measure: A callable returning one median, in milliseconds. Typically a closure over
            ``benchmark_single(...).median_ms``.
        samples: How many independent measurements to take.

    Returns:
        The list of medians, in call order. Order is preserved rather than sorted so a caller can
        see a monotone trend -- which would mean the device was still warming and the harvest is
        not yet a steady-state one.
    """
    return [measure() for _ in range(samples)]


#: Floor and ceiling for :func:`rounds_for`. The ceiling is `bench_timing.benchmark_extrapolated`'s
#: own default, so a cheap cell is budgeted to EXACTLY what it does today and nothing changes where
#: the current count is already affordable -- a re-budgeting that moved every cell would have to be
#: validated at every cell.
#:
#: The floor is 5 because that is the smallest count at which the inner median is still a median: it
#: survives two outlying windows, where 3 survives one and 1 is not a median at all. It is not
#: derived from a measurement, and it is not asserted to be safe -- K4 in `docs/fix_test_setup.md`
#: validates it by re-harvesting an expensive cell at both counts and requiring the medians to agree
#: within that cell's own band. If they do not, the floor is wrong and the item is withdrawn.
_MIN_ROUNDS, _MAX_ROUNDS = 5, 35

#: The round counts :func:`rounds_for` may return. Quantized, and snapped DOWN (to the largest rung
#: not exceeding the budgeted count) so the budget stays an upper bound.
#:
#: **Rungs alone do NOT prevent the count from oscillating, and an earlier version of this comment
#: claimed they did.** The count is derived from the cell's PINNED median and each harvest rewrites
#: that median, so a cell whose median sits near a rung BOUNDARY still flips: measured by the
#: property test, 23.292 ms gives 25 rounds and 20.963 ms gives 35 -- a 10% wobble across the
#: 22.9 ms boundary. Quantizing turns a continuum of boundaries into five, which makes the problem
#: rarer and not absent. What actually removes it is the HYSTERESIS in :func:`rounds_for_cell`,
#: which keeps a recorded count until the derived one is more than one rung away.
#:
#: The rungs are still load-bearing for that: hysteresis is defined in rung STEPS, so consecutive
#: ratios of 1.39x-1.6x mean a recorded count survives a median change of up to ~1.8x. Cells here
#: reproduce to 0.5-5%.
_ROUND_RUNGS = (5, 8, 12, 18, 25, 35)


def rounds_for(
    pin_median_ms: float,
    *,
    budget_s: float,
    launches_per_round: int,
    samples: int = N_SAMPLES_PIN_TIME,
) -> int:
    """How many timed rounds one harvest sample should use, given what the cell already costs.

    Purpose
        A harvest fires a FIXED launch count at every cell, and that count does not scale with how
        expensive the cell is. At `rounds=35`, `probe_iters=(1,4)` and `warmup=5` one
        `benchmark_extrapolated` call is `5*4 + 35*5 = 195` launches, so a pin sample of 15 costs
        2925 launches -- 11 s of device on a 1.76 ms cell, and 717 s on a 245 ms one. The second is
        thirteen minutes of re-measuring a number the repo's own calibration notes reproduce to
        0.6%. This converts that fixed count into a time budget.

    Semantics
        Pure arithmetic on the pin. **No measurement, no probing and no adaptivity** -- which is the
        load-bearing property, not an implementation detail: an adaptive per-rank repetition count
        is exactly what ``mode="device"`` does and why CLAUDE.md bans it for anything with a
        collective, since ranks then run different counts and DESYNC. Derived from bytes every rank
        already has, every rank computes the same count, by construction rather than by agreement.

        **Why ``rounds`` and not ``samples``.** The band a gate enforces is derived from ``rel_std``,
        the spread ACROSS the `samples` independent medians; `rounds` is the median WITHIN one of
        them and does not enter the band's width at all. So cutting `rounds` makes each sample
        noisier, which INFLATES `rel_std`, which WIDENS the band. That direction is the safe one: it
        can weaken a gate but it cannot manufacture a false failure. Cutting `samples` would do the
        opposite -- fewer observations of the same spread, an under-determined `rel_std`, and a band
        set from noise. That asymmetry is the whole reason this function budgets the term it does.

    Args:
        pin_median_ms: The cell's already-pinned median, in milliseconds. A re-harvest has this in
            hand before the cell runs. Non-finite or non-positive means "no pin" and yields
            `_MAX_ROUNDS` -- a FIRST harvest has nothing to budget from, and inventing a figure from
            a probe would reintroduce the adaptivity this design exists to avoid.
        budget_s: Target device seconds for the whole cell (all `samples`). Must be > 0; a
            non-positive budget would ask for zero rounds, and silently clamping that to the floor
            would hide a caller passing the wrong units.
        launches_per_round: Launches one timed round issues. For
            `benchmark_extrapolated(probe_iters=(n1, n2))` that is `n1 + n2`; for
            `benchmark_single(iters=N)` it is `N`. Must be >= 1. Passed rather than assumed because
            the two timers differ here and a wrong value scales the whole budget.
        samples: Independent medians the harvest takes. Defaults to `N_SAMPLES_PIN_TIME`, the count
            the band arithmetic assumes; pass another only if the caller also changed that.

    Returns:
        One of `_ROUND_RUNGS`, the largest not exceeding the budgeted count. Quantized so that a
        cell near a boundary cannot derive a different count on each harvest and demand a re-harvest
        forever -- see `_ROUND_RUNGS`. **The floor OVERRIDES the budget**, so a cell dear
        enough still exceeds it -- at 245 ms and a 60 s budget the answer is 5 rounds and 92 s. The
        budget is a target, and the floor is the statement that below it the inner median stops
        being one; honouring the budget instead would mean returning a count that measures nothing.
        Integer rather than float so two ranks cannot
        round the same value differently -- the arithmetic above is IEEE-deterministic for identical
        inputs on identical binaries, and every rank of a job is both.

    Raises:
        ValueError: `budget_s <= 0` or `launches_per_round < 1`, naming the value. Refused rather
            than clamped: both mean the caller's cost model is wrong, and a clamp would return a
            plausible count that silently does not honour the budget.
    """
    if budget_s <= 0:
        raise ValueError(f"budget_s must be > 0 seconds; got {budget_s!r}")
    if launches_per_round < 1:
        raise ValueError(f"launches_per_round must be >= 1; got {launches_per_round!r}")
    if not math.isfinite(pin_median_ms) or pin_median_ms <= 0:
        return _MAX_ROUNDS
    per_round_s = samples * launches_per_round * pin_median_ms * 1e-3
    raw = max(_MIN_ROUNDS, min(_MAX_ROUNDS, int(budget_s / per_round_s)))
    return max(r for r in _ROUND_RUNGS if r <= raw)


def rounds_for_cell(pins, key: Sequence, *, budget_s: float, launches_per_round: int) -> int:
    """The timed-round count for ONE cell, derived from that cell's own pin.

    Purpose
        The call a gate makes before building its ``measure`` closure, so the closure and the pin
        agree about the count. Splitting it from :func:`rounds_for` gives the gates one line instead
        of a pin lookup plus an arithmetic call, and puts the never-pinned fallback in exactly one
        place rather than in each gate.

    Semantics
        Looks the cell up and hands its pinned median to :func:`rounds_for`. **An unpinned cell gets
        the FLOOR**, because a first harvest has nothing to budget from and its only job is to
        produce a median good enough to choose a rung; inventing a figure from a probe instead would
        make the count per-rank, which is the desync `mode="device"` is banned for.

        A never-pinned CHEAP cell therefore needs two harvest passes -- seed at the floor, then
        re-measure at the count its own median derives, so `rel_std` and the count describe the same
        thing. An expensive one converges in ONE, because its budgeted count already IS the floor.
        That is the useful way round: the two-pass cost lands on the cells where a pass is cheap.

    Args:
        pins: the gate's :class:`pins.PinFile`.
        key: the cell's identity, as :func:`assert_cell` takes it.
        budget_s: target device seconds for the whole cell. See :func:`rounds_for`.
        launches_per_round: launches one timed round issues -- ``n1 + n2`` for
            ``benchmark_extrapolated(probe_iters=(n1, n2))``, ``iters`` for ``benchmark_single``.

    Returns:
        One of `_ROUND_RUNGS`. Identical on every rank given the same pin file, by construction.

        **Hysteretic, asymmetrically**: a recorded count is KEPT when the derived one is at most
        one rung BELOW it, and always yielded to when the derived one is higher. Without any
        hysteresis, a cell whose median sits near a rung boundary flips on every harvest and, since
        `assert_cell` refuses a mismatch, demands a re-harvest forever. Without the ASYMMETRY, a cell
        seeded at the floor would be held there forever, because its real budget is one rung away.
        Anchored on the recorded value rather than on any local state, so it is still a pure function
        of the pin file, and it reaches a fixed point within two harvests.
    """
    # `__contains__` / `__getitem__`, NOT `.get` -- `PinFile` has no `get`, and a `hasattr` probe
    # for one silently returns "unpinned" for EVERY cell, which reads as "the budget never applies"
    # and makes this whole item inert while every test still passes. Caught by writing the probe
    # first and checking the class second; it is pinned by
    # `test_the_round_count_is_read_through_PinFiles_REAL_lookup_api`.
    k = tuple(key)
    if k not in pins:
        # SEED AT THE FLOOR, not the ceiling, and this is a cost decision with a measured size.
        # A never-pinned cell has nothing to budget from, so it needs a first pass whose only job is
        # to produce a median good enough to CHOOSE A RUNG -- and rungs are 1.4x-1.6x apart, while a
        # 5-round median sits within a few percent of a 35-round one. Seeding at the ceiling instead
        # pays 7x for that estimate and then re-harvests anyway.
        #
        # For the cells this matters on it converges in ONE pass, not two: an expensive cell's
        # budgeted count IS the floor, so the seed's recorded count already matches what the seed's
        # own median derives. Measured on the N=8192 ladder, a 434 ms cell: ceiling-seed is
        # 21.4 + 5.1 = 26.5 min across two passes, floor-seed is 5.1 min in one. 5.2x.
        return _MIN_ROUNDS
    pin = pins[k]
    derived = rounds_for(pin.median_ms, budget_s=budget_s, launches_per_round=launches_per_round)
    # HYSTERESIS. Without it a cell whose median sits near a rung boundary derives one count on this
    # harvest and another on the next, and since `assert_cell` REFUSES a pin whose recorded count
    # differs from the derived one, that cell demands a re-harvest forever -- each one moving the
    # median just enough to move the count again. Measured by the property test before this existed:
    # 23.292 ms -> 25 rounds, 20.963 ms -> 35, a 10% wobble across the 22.9 ms boundary.
    #
    # Anchored on the RECORDED count, which every rank reads from the same bytes, so this stays a
    # pure function of the pin file and remains rank-uniform. One rung of slack means a recorded
    # count survives a median change of up to ~1.8x; beyond that the cell really has changed and the
    # count should move with it.
    recorded = getattr(pin, "rounds", None)
    if recorded in _ROUND_RUNGS:
        drop = _ROUND_RUNGS.index(recorded) - _ROUND_RUNGS.index(derived)
        # ASYMMETRIC: damp a DROP of one rung, always allow a RISE. Symmetric hysteresis was tried
        # first and is wrong once the seed is the floor -- a cell seeded at 5 whose real budget is 8
        # would be held at 5 forever, because 8 is one rung away. The seed's count is an artifact of
        # having no pin, not a considered choice, so it must not anchor the cell.
        #
        # Damping only the drop is also the right asymmetry on its own merits: rising means MORE
        # rounds, i.e. a better inner median and a better-measured band, and it is idempotent -- the
        # count settles at most one rung high and stops. Falling is the cost-cutting direction, and
        # it is where a boundary-straddling cell would otherwise demand a re-harvest on every run.
        if 0 <= drop <= 1:
            return recorded
    return derived


def harvest(measure: Callable[[], float], flops: float = 0.0) -> Stats:
    """Take a full pin-time sample of one thing and reduce it to what a gate pins.

    Purpose
        The ONE way to produce a `_PINS` entry, so a harvest cannot accidentally use a different
        sample count than the band arithmetic assumes. Before this existed the loop lived in an
        ad-hoc script and the printed output carried only a median -- which is exactly half of what
        a calibrated pin needs, and the half that cannot be recovered later.

    Args:
        measure: A callable returning one median in milliseconds.
        flops: The cell's work per call, reported as a rate only. 0 (the default) is the
            reference, which is cheap and always gets the full count.

    Returns:
        The :class:`Stats` to paste into the gate.
    """
    return summarize(repeat_median(measure, N_SAMPLES_PIN_TIME))


#: This session's reference median, set once by the ``reference_now`` fixture. Module state rather
#: than a fixture argument threaded through every gate: the drift factor describes the MACHINE, so
#: every gate in a run must divide by the same number, and passing it by hand through four files'
#: helpers is four chances for one of them to be left out and silently compare uncalibrated.


def assert_cell(
    measure: Callable[[], float],
    key: Sequence,
    pins,
    module_file: str,
    *,
    flops: float = 0.0,
    describe: Optional[Callable[[float], str]] = None,
    emit: bool = True,
    estimator: str = pins_mod.DEFAULT_ESTIMATOR,
    rounds_used: Optional[int] = None,
    cross_session_rel: float = 0.0,
) -> None:
    """Measure one cell and either harvest its pin or gate on it, drift-corrected.

    Purpose
        The ONE way a timed gate compares a measurement to a pin. Four gates previously wrote four
        versions of this block, and they had already diverged -- two used a size-dependent tolerance
        function, one a flat 10% constant, one the calibrated band -- so "how strict is this gate"
        was a per-file question. Routing all of them through here makes the answer uniform and makes
        a fifth variant something you have to work at.

    Semantics
        Three paths, chosen by environment and by whether the cell is pinned:

        * ``CPO_PERF_MEASURE=1`` -- takes the FULL :func:`harvest` sample and emits a machine-readable
          line via :func:`pins.emit_harvest`, then returns without asserting. The initial single
          measurement is still taken first, so ``describe`` reports a real number and the harvest
          measures a warm cell.
        * **unpinned** -- prints and SKIPS. A brand-new cell must not turn a suite red before anyone
          has had the chance to harvest it; the print is what they harvest from.
        * **pinned** -- compares directly against the pin and its band. No drift correction: see
          :func:`band_for` for why one made things worse, and :func:`clock_advice` for what
          carries the machine-state information instead.
          the pin's own band. A uniformly slower machine cancels instead of reading as a regression.

        The comparison is ONE-SIDED on purpose. A cell that comes in far FASTER than its pin is not
        failed here: that is either a real improvement or a measurement that did not run, and
        conflating the two into a red test trains people to re-pin without looking.

    Args:
        measure: Zero-argument callable returning ONE median in ms, via ``bench_utils``. Must be
            re-callable -- harvest mode invokes it repeatedly -- and must not synchronize or
            re-compile, since a JIT compile inside it would land in the timed region and dominate.
        key: The cell's identity. Must match the pin file's ``key`` exactly, including element
            types: ``(4096, 2048)`` and ``("4096", "2048")`` are different cells, and a mismatch
            reads as "unpinned" and SKIPS rather than failing.
        pins: The gate's :class:`pins.PinFile`. Passed rather than imported so this stays testable
            without a real file on disk.
        module_file: The gate's ``__file__``; tells :func:`pins.emit_harvest` which pin file the
            harvested line belongs to. A literal path here would send one gate's harvest into
            another gate's file.
        flops: Work per call. Used for the reported rate and to pick the harvest sample count -- a
            informational only -- the sample count is universal, so this never affects the gate.
        describe: Optional ``median_ms -> str`` appended to the printed line, for a gate that wants
            to report GB/s or TFLOP/s. Informational only; the gate is on wall time, never on the
            derived rate.
        emit: Whether harvest mode may write this measurement back to ``key``'s pin. MUST be False
            when the cell being measured is not the cell being compared against -- the autotune
            checks time a TUNED call against the FIXED default's pin, and emitting there would
            overwrite the fixed cell's pin with a tuned number, silently making the gate compare
            the tuner against itself.
        rounds_used: The timed-round count the caller's ``measure`` closure was built with, from
            :func:`rounds_for_cell`. Harvest mode RECORDS it on the pin; gate mode REFUSES a pin
            whose recorded count differs. None disables the check entirely, which is correct for a
            gate that has not adopted budgeted rounds -- the count is then whatever that gate's
            timer defaults to, and it is the same on both sides because nothing derives it.

    Returns:
        None.

    Raises:
        AssertionError: If the drift-corrected median exceeds the pin by more than its band.
        ValueError: If the cell is pinned but carries no ``rel_std``. That combination means the
            gate declared itself calibrated while its pin file was only half-harvested, and
            silently falling back to a flat tolerance is how a gate stops checking what it claims.
        pytest.skip.Exception: If the cell is unpinned.
    """
    import os

    import pytest

    from tests.perf.pins import emit_harvest

    if os.environ.get("CPO_PERF_MEASURE", "0") == "1":
        if not emit:
            one = measure()
            print(f"    {tuple(key)}: measured {one:.5f} ms (not harvested)")
            return
        st = harvest(measure, flops)
        note = f"  {describe(st.median_ms)}" if describe else ""
        emit_harvest(module_file, key, st.median_ms, st.rel_std, rounds_used)
        print(f"    {tuple(key)}: median={st.median_ms:.5f} rel_std={st.rel_std:.4f}{note}")
        return
    # The unpinned case is decided BEFORE the expensive sampling, and takes ONE sample rather than
    # `N_SAMPLES_TEST_TIME` of them. There is nothing to compare an unpinned cell against, so the
    # other 8 samples bought only a tighter estimate of a number that is about to be printed and
    # skipped. The single sample keeps that print -- it is what tells you what to harvest -- at 1/9
    # the cost.
    #
    # Measured cost of the old order, on the first run of the distributed A2A gate at 8 ranks: with
    # an empty pin file EVERY cell paid 9 x (15 rounds + 3 warmup) event-timed windows, each round
    # carrying an all_reduce, and the wedge watchdog aborted the job mid-suite. A gate whose pins
    # are not yet harvested is exactly the state a gate is authored in, so it must be cheap.
    if tuple(key) not in pins:
        one = measure()
        note = f"  {describe(one)}" if describe else ""
        print(f"    {tuple(key)}: {one:.5f} ms{note}")
        pytest.skip(
            f"no pin for {tuple(key)}; harvest one with "
            f"`CPO_PERF_MEASURE=1 pytest tests/perf/ -q -s`. The `-s` is REQUIRED: without it "
            f"pytest captures the PINHARVEST lines of passing tests and the log has none."
        )
    pin = pins[key]
    if pin.rel_std is None:
        raise ValueError(
            f"{tuple(key)} is pinned at {pin.median_ms} but carries no rel_std, so no band can be "
            f"derived. Re-harvest with `CPO_PERF_MEASURE=1 pytest tests/perf/ -q -s` (the `-s` is "
            f"required, or the harvest lines are captured); a pin file must be calibrated "
            f"as a whole, because a run that corrected some cells for drift and not others would "
            f"produce numbers that are not comparable with each other."
        )
    # The pin file must have been produced by the SAME estimator the caller is measuring with.
    # `benchmark_single(iters=20)` reads `device + C/20`; `benchmark_extrapolated` solves that term
    # out and reports `device`. Measured across the 13 pinned cp8 cells, 11 of 13 extrapolated
    # readings landed BELOW their iters=20 pin, mean ratio 0.989 -- small, ONE-SIDED, and on a
    # one-sided gate that reads as "everything got slightly faster" forever rather than as a
    # mistake. Refused per FILE because a file is calibrated as a whole.
    file_estimator = getattr(pins, "estimator", pins_mod.DEFAULT_ESTIMATOR)
    if file_estimator != estimator:
        raise ValueError(
            f"{tuple(key)}: this gate measures with estimator {estimator!r} but its pin file was "
            f"harvested with {file_estimator!r}. The two do not report the same quantity, and the "
            f"difference is one-sided, so comparing them would quietly loosen the gate rather than "
            f"fail it. RE-HARVEST the whole file with `CPO_PERF_MEASURE=1 pytest <this test> -q -s`; "
            f"a file must be calibrated as a whole."
        )
    # The pin must also have been harvested with the SAME timed-round count this gate is measuring
    # with, and the reason is the mirror of the estimator check above rather than a repeat of it.
    # `rounds` does not change what is being measured; it changes how ROBUST each sample's median is.
    # The band, though, is derived from `rel_std` -- the spread ACROSS samples, measured at harvest.
    # So a pin harvested at 35 rounds and gated at 5 leaves the band encoding 35-round noise while
    # the gate's samples carry 5-round noise: the band does NOT widen, the samples get noisier, and
    # the gate produces FALSE FAILURES. That is the opposite of the direction that makes budgeting
    # `rounds` safe at all, and it is the DEFAULT sequence, not an edge case -- a first harvest has
    # no pin so it runs at the ceiling, and every later run would derive a smaller count from the
    # pin it produced.
    #
    # A pin with no recorded count means 35, the historical default, because every pin in this repo
    # predating the field was harvested at 35. Reading it as "unknown, skip the check" would exempt
    # exactly the cells the check exists for.
    if rounds_used is not None:
        pinned_rounds = _MAX_ROUNDS if pin.rounds is None else pin.rounds
        if pinned_rounds != rounds_used:
            raise ValueError(
                f"{tuple(key)}: this gate measures with rounds={rounds_used} but the pin was "
                f"harvested with rounds={pinned_rounds}"
                f"{' (no recorded value; read as the historical default)' if pin.rounds is None else ''}"
                f". The band comes from the harvest's ACROSS-sample spread, so gating with fewer "
                f"rounds than the pin was taken with makes the samples noisier without widening the "
                f"band -- false failures, not a looser gate. RE-HARVEST this cell: "
                f"`CPO_PERF_MEASURE=1 pytest <this test> -q -s`."
            )
    cell = Stats(pin.median_ms, pin.rel_std)
    band = band_for(cell, cross_session_rel=cross_session_rel)
    # Escalating sampling: stop as soon as the cell is CLEARLY inside its band, otherwise keep
    # sampling up to the full count and judge on that -- see `_ESCALATION`. The loop always ends
    # with `median_ms` a median over however many samples were taken, so the failing path is
    # byte-for-byte the old decision rule.
    samples: list[float] = []
    early = cell.median_ms * (1.0 + (band - 1.0) * _EARLY_EXIT_FRAC)
    for target in _ESCALATION:
        while len(samples) < target:
            samples.append(measure())
        median_ms = statistics.median(samples)
        if median_ms <= early:
            break
    note = f"  {describe(median_ms)}" if describe else ""
    assert median_ms <= cell.median_ms * band, (
        f"{tuple(key)}: {median_ms:.5f} ms is {median_ms / cell.median_ms:.3f}x the pinned "
        f"{cell.median_ms:.5f} ms.{note} Band {band:.3f}x, from this cell's own measured spread "
        f"({cell.rel_std:.1%}) -- not a constant.\n"
        f"{clock_advice()}"
    )


def assert_host_dispatch(
    fn: Callable[[], object],
    key: Sequence,
    pins,
    module_file: str,
    *,
    device=None,
) -> None:
    """Gate ONE entry point's per-call host dispatch cost against its pin, CPU-corrected.

    Purpose
        The companion to :func:`assert_cell`, and the reason the two exist separately. A cell timed
        with ``mode="device"`` reports the kernel and says nothing about the submit path; this
        reports the submit path and says nothing about the kernel. Before the split, a launch-bound
        cell silently reported the second as if it were the first -- an 8x8 GEMM read 27 us of
        "kernel time" against 7 us of real device time, so a 20%% mainloop regression at small
        shapes was undetectable and a host change looked like a kernel change.

    Semantics
        Measures this entry's dispatch and the bare-``GemmSm90`` reference's BACK-TO-BACK with
        :func:`host_dispatch_us` (dual loop count, backpressure-guarded) and pins their RATIO. The
        ratio is what makes a dispatch pin portable: it cancels the CPU and leaves the repo's own
        per-call Python, which is the part a change to this repo can regress.

        **No drift factor, unlike the device gate.** Pairing the two halves in time is what replaces
        it -- and it is not an optimization but a correctness fix. Pinning absolute microseconds and
        dividing by a session-scoped host reference failed 2 runs in 4, with the factor reading
        0.86-0.95: always fast, never slow. The bias lived in the PIN, whose reference half was
        harvested at session start and whose target half minutes later. A ratio measured
        back-to-back has no moment at which only one half was measured, so there is nothing left for
        a correction to correct.

        Harvest mode takes the full pin-time sample and emits, like every other pinned number.
        One-sided, like the device gate: dispatching FASTER than pinned is never a failure.

    Args:
        fn: Zero-argument callable issuing exactly one call through the entry point being gated,
            at a TINY shape. Must already have been called once (JIT compile outside the loop) and
            must not synchronize. A large shape does not break correctness here but wastes time,
            and past ~1000 pending launches would invite the backpressure the guard refuses.
        key: The dispatch pin's identity, e.g. ``("dispatch", "gemm")``. Lives in the same pin file
            as the cells and in the same ``cells`` list, distinguished only by its key, so nothing
            in the schema had to grow a second shape.
        pins: The gate's :class:`pins.PinFile`.
        module_file: The gate's ``__file__``, for :func:`pins.emit_harvest`.
        device: Device to drain around the timed span; None uses the current device.

    Returns:
        None.

    Raises:
        AssertionError: If corrected dispatch exceeds the pin by more than its band.
        ValueError: If pinned without a ``rel_std``.
        pytest.skip.Exception: If unpinned.
        RuntimeError: From :func:`host_dispatch_us` if the reading is backpressure-tainted.
    """
    import os

    import pytest

    from tests.perf.pins import emit_harvest

    def paired_ratio():
        """One PAIRED sample: this entry's dispatch divided by the reference's, measured adjacent.

        The ratio is formed from two measurements taken back-to-back, so whatever the CPU is doing
        at that moment is common-mode and cancels. That is the whole reason a ratio is pinned here
        rather than an absolute microsecond count with a separate drift correction.
        """
        target = host_dispatch_us(fn, device)
        ref = host_dispatch_us(_host_reference(device), device)
        return target / ref

    if os.environ.get("CPO_PERF_MEASURE", "0") == "1":
        st = harvest(paired_ratio)
        emit_harvest(module_file, key, st.median_ms, st.rel_std)
        print(f"    {tuple(key)}: dispatch {st.median_ms:.3f}x reference rel_std={st.rel_std:.4f}")
        return
    now = statistics.median(repeat_median(paired_ratio, N_SAMPLES_TEST_TIME))
    if tuple(key) not in pins:
        print(f"    {tuple(key)}: dispatch {now:.3f}x reference")
        pytest.skip(
            f"no dispatch pin for {tuple(key)}; harvest with "
            f"`CPO_PERF_MEASURE=1 pytest tests/perf/ -q -s` (the `-s` is required)"
        )
    pin = pins[key]
    if pin.rel_std is None:
        raise ValueError(
            f"{tuple(key)} is pinned at {pin.median_ms:.3f}x but carries no rel_std, so no band can "
            f"be derived. Re-harvest with `CPO_PERF_MEASURE=1 pytest tests/perf/ -q -s`."
        )
    cell = Stats(pin.median_ms, pin.rel_std)
    # No drift factor: the pinned quantity is ALREADY normalized, because both halves of the ratio
    # were measured at the same moment. An earlier version pinned absolute microseconds and divided
    # by a session-scoped reference, and the two halves of that implied ratio came from different
    # moments -- the reference from session start, the target from minutes later. On this CPU that
    # was a ONE-SIDED 7-14% bias (drift read 0.86-0.95, never above 1), which failed a target
    # sitting 1.6% from its pin. Re-measuring the reference adjacent did not fix it, because the
    # PIN carried the same mismatch; pinning the ratio does, because there is no longer a moment at
    # which only one half was measured.
    # `abs_jitter_ms=0.0` because this pin is a dimensionless RATIO of two host costs, not a
    # duration: an absolute-millisecond jitter has no meaning divided by 1.447. The ratio is also
    # the one quantity here a slower machine cannot move, so its noise is genuinely proportional
    # and the pinned rel_std is the right and only term.
    band = band_for(cell, abs_jitter_ms=0.0)
    assert now <= cell.median_ms * band, (
        f"{tuple(key)}: host dispatch is {now:.3f}x a bare GemmSm90 launch, against the pinned "
        f"{cell.median_ms:.3f}x -- {now / cell.median_ms:.3f}x of it, band {band:.3f}x. This ratio "
        f"is the repo's PER-CALL PYTHON cost (epilogue args, scheduler args, data_ptr(), the "
        f"jit_cache lookup, the autotune gate) in units that cancel the CPU, so a slower machine "
        f"cannot cause this. Something was added to the submit path."
    )
