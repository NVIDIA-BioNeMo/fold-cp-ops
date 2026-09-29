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

"""Tests for ``fold_cp_ops._internal.autotune.timing`` -- how a candidate is measured.

The module's whole job is to make one mistake unreachable, and the tests are shaped around it:
**an adaptive timer desynchronizes collectives.** A timer that decides its repetition count from a
first estimate gives different counts on different ranks; the ranks then execute different numbers
of collectives, and the fast one exits its loop while the slow one is still inside a collective
waiting for it. That is a hang, minutes later, with nothing pointing back here.

So ``mode`` is not a knob -- it is a request that gets REFUSED. Accepting-and-ignoring it is how the
wrong timer survives review, which is exactly what happened upstream, where the default was
``triton.testing.do_bench`` for every kernel including the distributed ones.
"""

import pytest

from fold_cp_ops._internal.autotune.timing import ITERS, ROUNDS, WARMUP, TimingPolicy


def test_the_mode_is_derived_from_whether_a_collective_is_present():
    """Local tunes on DEVICE time; a collective tunes on the event window. Never the reverse.

    The two constraints point opposite ways and both are hard:

    * **Collective must be ``event``.** The adaptive timer picks its repetition count per rank, so
      ranks run different numbers of collectives and desynchronize.
    * **Local must be ``device``.** The event window measures the cadence of a launch stream, and at
      a launch-bound shape that cadence is the ~22 us host submit floor rather than the kernel -- a
      128x256x256 GEMM reads 27.3 us of "kernel time" against 7.2 us of device time. Every candidate
      then measures the same floor and the sweep's winner is noise. Observed at
      ``(1001, 2048, 2048, 1)``: two runs picked two different configs, 1.11x and 1.25x worse than
      the best known.

    Device mode is safe locally for precisely the reason it is unsafe distributed -- with one rank
    there is nothing to stay in lockstep with.
    """
    assert TimingPolicy().mode == "device"
    assert TimingPolicy(collective=True).mode == "event"


@pytest.mark.parametrize(
    "collective,asked,message",
    [(False, "event", r"host submit floor"), (True, "device", r"desynchronize")],
)
def test_the_wrong_mode_is_refused_rather_than_downgraded(collective, asked, message):
    """Asking for the wrong timer raises, and the message says which failure it would cause.

    Silently substituting the right one would be worse than either alternative: the caller believes
    they got what they asked for, and the next person to want that mode has no evidence it was ever
    considered. Both directions are covered because they are different bugs -- one desynchronizes a
    distributed sweep, the other makes a local sweep pick noise -- and a refusal that only knew
    about the first is what let the second ship.
    """
    with pytest.raises(ValueError, match=message):
        TimingPolicy(collective=collective).measure(lambda: None, mode=asked)


def test_an_injected_double_without_a_mode_parameter_still_works():
    """The injection contract is the fixed-iteration semantics, not a signature.

    Doubles predate the `mode` parameter. Passing it unconditionally would break every one of them
    with a TypeError that says nothing about what changed, so the policy passes it only to a
    callable that accepts it.
    """
    seen = {}

    def old_style(fn, *, rounds, warmup, iters, reduce):
        """A double written against the pre-`mode` interface."""
        seen.update(rounds=rounds, warmup=warmup, iters=iters, reduce=reduce)
        return 1.0

    assert TimingPolicy(measure=old_style).measure(lambda: None) == 1.0
    assert seen == {"rounds": ROUNDS, "warmup": WARMUP, "iters": ITERS, "reduce": None}


def test_a_mode_aware_double_receives_the_derived_mode():
    """A double that DOES accept `mode` is told which one, so it can assert on it."""
    seen = {}

    def new_style(fn, *, rounds, warmup, iters, reduce, mode):
        """A double written against the current interface."""
        seen["mode"] = mode
        return 1.0

    TimingPolicy(measure=new_style).measure(lambda: None)
    assert seen["mode"] == "device"
    TimingPolicy(collective=True, measure=new_style).measure(lambda: None)
    assert seen["mode"] == "event"


def test_the_measurement_geometry_is_fixed_not_adaptive():
    """The rounds/warmup/iters constants are module-level and are passed through unchanged.

    Fixed is the point: a per-rank repetition count is the desync. They are asserted here as
    constants rather than trusted, because a "small tuning tweak" that made them adaptive would be
    a one-line change with a hang as its symptom.
    """
    seen = {}

    def fake(fn, *, rounds, warmup, iters, reduce):
        """Record the geometry the policy asked for, and return a fixed time."""
        seen.update(rounds=rounds, warmup=warmup, iters=iters, reduce=reduce)
        return 1.5

    assert TimingPolicy(measure=fake).measure(lambda: None) == 1.5
    assert seen["rounds"] == ROUNDS and seen["warmup"] == WARMUP and seen["iters"] == ITERS
    assert all(isinstance(v, int) for v in (ROUNDS, WARMUP, ITERS))


@pytest.mark.parametrize("collective,expected", [(False, None), (True, "max")])
def test_the_reduction_is_derived_from_whether_a_collective_is_present(collective, expected):
    """``reduce="max"`` under a collective, nothing without one -- and it is not a caller's choice.

    Getting this wrong in the False direction times a distributed kernel per-rank with no reduction:
    the numbers are each rank's own view, and the config that "wins" is whichever rank waited least.
    Deriving it from `collective` rather than exposing it is what makes that unreachable.
    """
    seen = {}

    def fake(fn, *, rounds, warmup, iters, reduce):
        """Record the reduction the policy asked for."""
        seen["reduce"] = reduce
        return 1.0

    TimingPolicy(collective=collective, measure=fake).measure(lambda: None)
    assert seen["reduce"] == expected


def test_the_default_timer_is_the_packaged_one():
    """With no injected timer the policy resolves `_default_measure`, never a hand-rolled loop.

    CLAUDE.md forbids rolling one: a ``perf_counter`` loop around an async launch measures host
    DISPATCH, not device time, and yields "bandwidths" above the physical link. The default used to
    be resolved lazily from ``benchmark/``, which is not packaged -- so on an installed wheel it
    raised. It now lives in the package.
    """
    from fold_cp_ops._internal.autotune.timing import _default_measure

    assert TimingPolicy()._resolve() is _default_measure
    injected = lambda fn, **kw: 1.0  # noqa: E731
    assert TimingPolicy(measure=injected)._resolve() is injected


def test_a_zero_warmup_skips_the_thermal_loop_entirely():
    """``warmup_ms=0`` must not touch the GPU at all, so the CPU-only tests here can run.

    The thermal warmup exists because the first config benchmarked otherwise runs on a cool part and
    wins on temperature rather than on merit. Being able to turn it off is what makes this module
    testable without a device.
    """
    TimingPolicy(warmup_ms=0).thermal_warmup()  # must not raise, must not allocate


def test_the_warmup_is_bracketed_by_the_barrier_it_is_given():
    """A supplied barrier is called BEFORE and AFTER, or the warmup becomes a measurement skew.

    An unsynchronized warmup means one rank is still saturating its GPU while another is timing --
    which makes the timing rank's collective look slow, and the config it was measuring lose.
    """
    calls = []
    policy = TimingPolicy(warmup_ms=0)
    policy.thermal_warmup(barrier=lambda: calls.append(1))
    assert calls == [], "a disabled warmup must not barrier either, or ranks desync on a no-op"


def test_verbose_reads_the_cpo_prefixed_variable(monkeypatch):
    """``CPO_AUTOTUNE_VERBOSE`` and nothing else. A half-renamed env var reads as 'off'."""
    monkeypatch.delenv("CPO_AUTOTUNE_VERBOSE", raising=False)
    assert TimingPolicy.verbose() is False
    monkeypatch.setenv("CPO_AUTOTUNE_VERBOSE", "1")
    assert TimingPolicy.verbose() is True
