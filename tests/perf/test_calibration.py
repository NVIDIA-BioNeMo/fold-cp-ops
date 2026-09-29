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

"""Tests for ``tests/perf/calibration.py`` -- the pinned band and the gate front doors.

The arithmetic here decides whether a perf gate fires, so it is tested on its own rather than only
through a gate: a band that silently came out too wide would make every downstream cell vacuous
while every downstream cell still passed. Almost all of it is pure functions of numbers, so almost
all of it runs without a GPU.

Deliberately NOT named ``test_benchmark_perf_*``: the isolation hook in ``conftest.py`` skips only
the timed gates, and these are not timed. They must run under ``pytest tests/`` like any other unit
test, because they are the thing that says the timed gates mean anything.
"""

import math

import pytest
from _pytest.outcomes import Skipped

from fold_cp_ops.testing.kernel_matrix import matrix_exempt

import tests.perf.pins as pins_mod
from tests.perf.calibration import (
    _ESCALATION,
    N_SAMPLES_PIN_TIME,
    N_SAMPLES_TEST_TIME,
    SIGMA,
    Stats,
    _ABS_JITTER_MS,
    _CROSS_SESSION_REL,
    _MIN_REL_STD,
    _STD_INFLATE,
    assert_cell,
    band_for,
    harvest,
    rounds_for,
    rounds_for_cell,
    repeat_median,
    summarize,
)
from tests.perf.pins import HARVEST_TAG


# ── the pinned pair ───────────────────────────────────────────────────────────────────────────
@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_a_pin_needs_a_positive_median_and_a_non_negative_spread():
    """A degenerate pin is refused where it is built, not where it is compared."""
    Stats(median_ms=1.0, rel_std=0.0)  # legal: a perfectly stable cell
    with pytest.raises(ValueError, match=r"median_ms must be positive"):
        Stats(median_ms=0.0, rel_std=0.1)
    with pytest.raises(ValueError, match=r"rel_std must be non-negative"):
        Stats(median_ms=1.0, rel_std=-0.1)


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_summarize_reports_the_median_and_the_relative_spread():
    """The spread is RELATIVE, so a pin re-harvested on a faster machine keeps its meaning."""
    s = summarize([10.0, 10.0, 10.0, 11.0, 9.0])
    assert s.median_ms == 10.0
    assert s.rel_std == pytest.approx(0.0707, abs=1e-3)  # stdev 0.707 over median 10


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_summarize_floors_a_suspiciously_stable_cell():
    """A zero-width band would fail on the first jitter, so the spread has a floor.

    This is not a fudge: a cell that measures identically five times has not proven it has no
    noise, it has proven the timer's resolution hid it.
    """
    assert summarize([5.0] * 6).rel_std == pytest.approx(0.005)


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_one_sample_is_refused():
    """A single observation carries no spread, and pretending otherwise pins a too-tight band."""
    with pytest.raises(ValueError, match=r"at least 2 samples"):
        summarize([1.0])


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_summarize_is_robust_to_a_single_outlier():
    """The MEDIAN sets the pin, so one slow sample moves the band and not the target.

    That asymmetry is deliberate: an outlier is evidence about noise, not about where the kernel
    sits, and letting it drag the pin upward would ratchet the gate looser on every re-harvest.
    """
    clean = summarize([10.0, 10.1, 9.9, 10.0, 10.05])
    spiked = summarize([10.0, 10.1, 9.9, 10.0, 10.05, 14.0])
    assert spiked.median_ms == pytest.approx(clean.median_ms, rel=0.01)
    assert spiked.rel_std > clean.rel_std * 3


# ── the band ──────────────────────────────────────────────────────────────────────────
@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_the_band_scales_with_the_cells_own_spread():
    """A noisy cell gets a wide band and a stable one a narrow band, with nobody choosing either.

    This is the whole point of pinning the spread: the old gate used one constant for cells whose
    real spreads differed by more than 10x, so it was simultaneously too tight for the launch-bound
    ones and too loose for the compute-bound ones.
    """
    narrow = band_for(Stats(1.0, 0.005))
    wide = band_for(Stats(1.0, 0.06))
    assert 1.0 < narrow < wide
    assert wide > 1.25 and narrow < 1.05


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_the_band_is_sigma_times_the_inflated_spread():
    """The formula is checked directly, so a change to it has to be deliberate.

    Uses a 1 ms cell so the absolute-jitter term is negligible in quadrature; the term's own
    behaviour is pinned by the two tests below rather than smeared through this one.
    """
    cell = Stats(1.0, 0.02)
    assert band_for(cell) == pytest.approx(1.0 + SIGMA * 0.02 * 1.25, rel=1e-3)


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_a_short_cell_gets_a_wider_band_than_a_long_one_at_the_SAME_spread():
    """Identical pinned spread, 100x different duration -- the short cell must be gated looser.

    The pinned ``rel_std`` is harvested within ONE process, so for a cell measured in microseconds
    it describes back-to-back agreement and says nothing about the across-session variation the gate
    actually faces. Measured: four whole-suite runs of two trees produced 28 distinct failing cells
    of which only 3 failed every time, and the SAME tree run twice gave 7 failures and then 20 --
    all of them cells of 9-16 us pinned at the 0.5% spread floor.

    Asserted as an inequality plus a magnitude, not against a constant, so re-measuring
    ``_ABS_JITTER_MS`` does not force an edit here -- but shrinking it to nothing does.
    """
    short = band_for(Stats(0.010, _MIN_REL_STD))
    long_ = band_for(Stats(1.000, _MIN_REL_STD))
    assert short > long_, "a 10 us cell must not be gated as tightly as a 1 ms one"
    assert short > 1.03, f"the widening must be worth having; got {short:.4f}"


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_a_compute_bound_cell_is_left_alone_by_the_widening():
    """The absolute term must not quietly loosen the cells that were never the problem.

    A millisecond-scale cell's band has to stay what its own measured spread says, or the widening
    buys stability at the cost of the gate's power where the gate still works. Quadrature is what
    makes that true for free: a sub-microsecond jitter against a 1 ms median is a part in a
    thousand, and squaring it puts it a part in a million below the spread term.
    """
    spread_only = 1.0 + SIGMA * 0.02 * _STD_INFLATE
    assert band_for(Stats(1.0, 0.02)) == pytest.approx(spread_only, rel=1e-4)


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_repeat_median_preserves_call_order():
    """Order is kept so a monotone trend is visible -- it means the device was still warming up.

    A sorted return would hide exactly the condition that invalidates a harvest.
    """
    seq = iter([3.0, 2.0, 1.0])
    assert repeat_median(lambda: next(seq), 3) == [3.0, 2.0, 1.0]


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_harvest_takes_the_pin_time_sample_count_and_returns_a_pin():
    """`harvest` is the ONE way to build a pin, so a hand-rolled loop cannot use a different count.

    Before it existed the loop lived in an ad-hoc script whose output carried a median only --
    exactly half of what a calibrated pin needs, and the half that cannot be recovered afterwards.
    """
    calls = {"n": 0}

    def measure():
        """Return a slightly different time each call, so the spread is non-degenerate."""
        calls["n"] += 1
        return 1.0 + 0.01 * (calls["n"] % 3)

    st = harvest(measure)
    assert calls["n"] == N_SAMPLES_PIN_TIME
    assert isinstance(st, Stats) and st.median_ms > 0 and st.rel_std > 0

    calls["n"] = 0
    harvest(measure, flops=1e15)
    assert calls["n"] == N_SAMPLES_PIN_TIME, "the count is universal: work does not change it"


# -- the shared gate front door -----------------------------------------------------------------
@pytest.fixture(autouse=True)
def _not_harvesting(monkeypatch):
    """Clear ``CPO_PERF_MEASURE`` so these tests describe the GATING path regardless of the caller.

    `assert_cell` reads the variable at call time, so without this every test below flips to the
    harvest path when someone runs the suite during a re-harvest -- which is exactly when this
    machinery is being changed and most needs checking. Measured: a harvest run turned 4 of these
    green-by-accident into failures, which is the honest version of the same problem.

    The one test that exercises harvest mode sets the variable itself, and a later ``setenv`` wins
    over this ``delenv``.

    Yields:
        None.
    """
    monkeypatch.delenv("CPO_PERF_MEASURE", raising=False)
    yield


class _FakePins:
    """A stand-in for ``pins.PinFile`` holding one cell, so these run without a file on disk."""

    def __init__(self, cells, estimator=None):
        """Store the mapping from tuple key to :class:`pins.Pin`-alike.

        ``estimator`` defaults to the module default, so every pre-existing construction of this
        fake keeps meaning "a file harvested the ordinary way".
        """
        self._cells = cells
        self.estimator = pins_mod.DEFAULT_ESTIMATOR if estimator is None else estimator

    def __contains__(self, key):
        """Whether the cell is pinned."""
        return tuple(key) in self._cells

    def __getitem__(self, key):
        """The pin for one cell."""
        return self._cells[tuple(key)]


class _FakePin:
    """A pin with the fields ``assert_cell`` reads."""

    def __init__(self, median_ms, rel_std, rounds=None):
        """Store the pinned median, its spread (None meaning un-harvested) and its round count.

        ``rounds`` defaults to None, which `assert_cell` reads as the historical 35 -- so every
        pre-existing construction of this fake keeps meaning "harvested the ordinary way".
        """
        self.median_ms, self.rel_std, self.rounds = median_ms, rel_std, rounds


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_a_cell_within_its_band_passes_and_one_beyond_it_fails():
    """The gate itself: a measurement inside the band passes, one beyond it fails."""
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.02)})
    band = band_for(Stats(1.0, 0.02))
    assert_cell(lambda: band * 0.99, (1, 2), pins, "/x/test_benchmark_perf_x.py")
    with pytest.raises(AssertionError, match=r"the pinned"):
        assert_cell(lambda: band * 1.01, (1, 2), pins, "/x/test_benchmark_perf_x.py")


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_the_gate_medians_several_samples_so_one_transient_cannot_fail_it():
    """A single excursion is discarded; a genuine regression in every sample still fails.

    Both halves are asserted because either alone is the wrong gate. Taking the MINIMUM would also
    survive the transient, but it would survive a real regression that happened to have one fast
    sample too. The median needs a majority to move it.

    The number this defends against is measured, not hypothetical: consecutive samples of the
    4096x2048x1024 epilogue cell agree to 0.6%, and that same cell has been seen ~8% high for a
    whole `benchmark_single` call. A single-sample gate cannot tell that from a regression.
    """
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01)})
    transient = iter([1.0] * 4 + [8.0] + [1.0] * (N_SAMPLES_TEST_TIME - 5))  # one wild sample
    assert_cell(lambda: next(transient), (1, 2), pins, "/x/test_benchmark_perf_x.py")
    with pytest.raises(AssertionError):
        assert_cell(lambda: 8.0, (1, 2), pins, "/x/test_benchmark_perf_x.py")


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_a_cell_well_inside_its_band_costs_ONE_sample():
    """The happy path stops at the first rung of `_ESCALATION`, not `N_SAMPLES_TEST_TIME`.

    This is the whole point of the ladder. A gate cell costs `samples x 43 windows x 20 iters`
    launches, and that count does not scale with how expensive the cell is -- so the workflow's
    61 ms cell spent 475 s of device time re-confirming a number one sample already reproduces to
    ~0.6% against a 2.5% band. Nine samples of a cell measuring 1.000x its pin buy nothing.

    Asserted as exactly 1, not "fewer than 9": a ladder that quietly settled on 3 would still look
    like an improvement while costing 3x what it needs to.
    """
    calls = {"n": 0}

    def measure():
        """Count invocations and return a value comfortably inside the band."""
        calls["n"] += 1
        return 1.0

    assert_cell(measure, (1, 2), _FakePins({(1, 2): _FakePin(1.0, 0.01)}), "/x/t.py")
    assert calls["n"] == _ESCALATION[0] == 1, f"expected one sample, took {calls['n']}"


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_a_cell_NEAR_its_band_escalates_to_the_full_sample_count():
    """A cell inside the band but past the early-exit threshold must still pay for all 9.

    THE control on the cheap path. Early exit is only sound because it is refused exactly where the
    answer is in doubt: a cell sitting just under its band is where one noisy sample can hide a real
    regression, and it is precisely the case a "take fewer samples" change would break silently --
    the suite would stay green while the gate quietly lost its power near the threshold.

    Band here is 1 + 4.0 * 0.01 * 1.25 = 1.05, so early exit needs <= 1.025. 1.04 is inside the
    band (it PASSES) but past the threshold, so it must escalate and be judged on the median of 9.
    """
    calls = {"n": 0}

    def measure():
        """Count invocations and return a value that passes but is NOT comfortably inside."""
        calls["n"] += 1
        return 1.04

    assert_cell(measure, (1, 2), _FakePins({(1, 2): _FakePin(1.0, 0.01)}), "/x/t.py")
    assert calls["n"] == N_SAMPLES_TEST_TIME, (
        f"a cell near its band must escalate to {N_SAMPLES_TEST_TIME} samples, took {calls['n']}"
    )


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_harvest_mode_does_not_pay_for_the_gates_samples():
    """Harvesting takes the pin-time count and NOT the gate's on top of it.

    An earlier version measured once for the report and then harvested, so every harvested cell paid
    for a sample nobody used. Worse, the two paths then measured different quantities, which is the
    kind of asymmetry that makes a pin and the gate reading it quietly incomparable.
    """
    calls = {"n": 0}

    def measure():
        """Count invocations; the value only needs to be non-degenerate."""
        calls["n"] += 1
        return 1.0 + 0.01 * (calls["n"] % 3)

    import os

    os.environ["CPO_PERF_MEASURE"] = "1"
    try:
        assert_cell(measure, (1, 2), _FakePins({}), "/x/test_benchmark_perf_x.py")
    finally:
        del os.environ["CPO_PERF_MEASURE"]
    assert calls["n"] == N_SAMPLES_PIN_TIME


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_a_slower_machine_FAILS_and_the_message_reports_the_clock_without_prescribing():
    """The deliberate trade: no drift correction, so machine state shows up as a failure.

    This replaces a test asserting the opposite -- that a uniform slowdown CANCELLED. It did, for a
    slowdown that hit the reference and the target equally, and almost nothing on real hardware
    does: unlocked, a compute-bound reference sped up 22% while bandwidth-bound LayerNorm sped up
    6.6%, and dividing one by the other produced 23 false failures in a single run. Correcting for
    the clock was never reliable; LOCKING it is, so the clock became a precondition.

    What replaces the correction is a MESSAGE. A ratio alone cannot distinguish "the kernel got
    slower" from "the box is in a different state", so the failure text has to, and that is the part
    worth pinning here -- an unlocked box producing bare ratios is how a batch of false failures gets
    read as a regression.
    """
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01)})
    with pytest.raises(AssertionError, match=r"the pinned") as exc:
        assert_cell(lambda: 1.12, (1, 2), pins, "/x/test_benchmark_perf_x.py")
    msg = str(exc.value)
    assert "SM clock at read time" in msg, "the reader needs the clock the measurement was taken at"
    assert "do NOT lock the clock" in msg, (
        "the message must actively warn AGAINST locking: pins are harvested free-running, so a "
        "locked re-run compares against a different distribution and is likelier to cause a "
        "failure than to explain one"
    )


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_a_faster_than_pinned_cell_is_not_failed():
    """One-sided by design: an improvement and a measurement that did not run look alike.

    Failing the fast side would train people to re-pin on every green-looking anomaly, which is how
    a pin ratchets to whatever the machine did last.
    """
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01)})
    assert_cell(lambda: 0.4, (1, 2), pins, "/x/test_benchmark_perf_x.py")


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_an_unpinned_cell_skips_and_prints_what_to_pin(capsys):
    """A new cell must not turn the suite red before anyone could have harvested it."""
    with pytest.raises(Skipped, match=r"CPO_PERF_MEASURE=1"):
        assert_cell(lambda: 1.0, (9, 9), _FakePins({}), "/x/test_benchmark_perf_x.py")
    assert "1.00000 ms" in capsys.readouterr().out


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_a_pin_without_a_spread_is_refused_rather_than_silently_flat_banded():
    """Half a harvest is the dangerous state: the gate looks calibrated and checks something else.

    Falling back to a flat tolerance here would let a gate declare `_CALIBRATED` on a pin file that
    cannot support it, and the failure would surface as a mysteriously loose band rather than as
    the missing harvest it is.
    """
    pins = _FakePins({(1, 2): _FakePin(1.0, None)})
    with pytest.raises(ValueError, match=r"no rel_std"):
        assert_cell(lambda: 1.0, (1, 2), pins, "/x/test_benchmark_perf_x.py")


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_harvest_mode_emits_a_machine_readable_line_and_asserts_nothing(capsys, monkeypatch):
    """Harvesting must not fail on the pin it is in the middle of replacing.

    The measurement here is 100x its pin: under the gating path that is a hard failure, and if
    harvest mode shared that path a re-harvest could never get past the first regressed cell --
    which is precisely when a re-harvest is needed.
    """
    monkeypatch.setenv("CPO_PERF_MEASURE", "1")
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01)})
    assert_cell(lambda: 100.0, (1, 2), pins, "/x/test_benchmark_perf_x.py")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(HARVEST_TAG)]
    assert len(lines) == 1 and "[1,2]" in lines[0]


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_emit_false_compares_against_a_pin_without_harvesting_over_it(capsys, monkeypatch):
    """A cell timed against ANOTHER cell's pin must never write back to it.

    The autotune checks time a TUNED call and gate it on the FIXED default's pinned time. Emitting
    there would replace the fixed pin with the tuned number, after which the gate compares the tuner
    against its own previous output and can never fail -- a gate that erases its own baseline.
    """
    monkeypatch.setenv("CPO_PERF_MEASURE", "1")
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01)})
    assert_cell(lambda: 0.5, (1, 2), pins, "/x/test_benchmark_perf_x.py", emit=False)
    out = capsys.readouterr().out
    assert HARVEST_TAG not in out and "not harvested" in out


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_the_describe_hook_annotates_the_line_without_entering_the_comparison():
    """A gate may report GB/s or TFLOP/s; the gate is still on wall time.

    Stated as a test because the tempting shortcut -- gating on the derived rate -- would make the
    threshold depend on the gate's own arithmetic for bytes or flops, so a mistake there would move
    the bar rather than the reported number.
    """
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01)})
    with pytest.raises(AssertionError, match=r"640 TFLOP/s"):
        assert_cell(
            lambda: 10.0,
            (1, 2),
            pins,
            "/x/test_benchmark_perf_x.py",
            describe=lambda _: "640 TFLOP/s",
        )


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_the_cross_session_term_is_off_unless_a_gate_opts_in():
    """A band measured on one venue must not silently widen every other gate.

    ``_CROSS_SESSION_REL`` was measured on the two-node A2A workflow gate, whose pin and run land on
    whatever venue A nodes Slurm hands out. Most gates in this repo pin and run on the SAME box in
    the same session, and for them that spread is not a noise source -- applying it would buy them
    nothing and cost them detection power. So the default must be zero, and this asserts it against
    the parameter being present at all: a default of ``_CROSS_SESSION_REL`` would type-check, pass
    every other test in this file, and quietly halve eleven gates' sensitivity.
    """
    cell = Stats(1.0, _MIN_REL_STD)
    assert band_for(cell) == pytest.approx(band_for(cell, cross_session_rel=0.0)), (
        "the default must be 0.0 -- an opt-in term that is on by default is not opt-in"
    )
    assert band_for(cell) == pytest.approx(1.0 + SIGMA * _MIN_REL_STD * _STD_INFLATE, rel=1e-3)


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_the_cross_session_term_widens_the_floor_and_fades_on_a_noisy_cell():
    """It must rescue the cells that were failing and leave the ones that were not.

    The false failures were at the ``_MIN_REL_STD`` floor -- cells whose within-process spread
    bottomed out, leaving the band no room for a machine change. Measured: two cells at rel_std=0.005
    came in at 1.036x and 1.026x against bands of 1.025x.

    A cell whose own spread already exceeds the term must barely move, which is what quadrature buys
    and a sum would not: at rel_std=0.06 the term contributes sqrt(0.06^2+0.01^2)/0.06 = 1.4%, so the
    genuinely noisy cells keep being gated at their own noise level rather than at this one.
    """
    floor = Stats(1.0, _MIN_REL_STD)
    widened = band_for(floor, cross_session_rel=_CROSS_SESSION_REL)
    assert widened > band_for(floor), "the term must widen the floor or it does nothing"
    assert widened > 1.036, f"must clear the measured 1.036x false failure; got {widened:.4f}"
    noisy = Stats(1.0, 0.06)
    assert band_for(noisy, cross_session_rel=_CROSS_SESSION_REL) < band_for(noisy) * 1.01, (
        "a cell already noisier than the term must be left essentially alone -- quadrature, not a sum"
    )


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_the_two_noise_terms_combine_in_quadrature_not_by_addition():
    """Both terms at once must add as variances, or a short cross-venue cell is over-widened.

    ``_ABS_JITTER_MS`` and ``_CROSS_SESSION_REL`` describe independent things -- a fixed launch-path
    jitter and a proportional machine difference -- so their variances add and their standard
    deviations do not. Summing would roughly double the widening on exactly the short cells that
    carry both, which is over-correcting by the amount that hides a real regression.
    """
    short = Stats(0.010, _MIN_REL_STD)
    got = band_for(short, cross_session_rel=_CROSS_SESSION_REL) - 1.0
    rel = math.hypot(_MIN_REL_STD, _ABS_JITTER_MS / short.median_ms, _CROSS_SESSION_REL)
    assert got == pytest.approx(SIGMA * rel * _STD_INFLATE, rel=1e-6)
    summed = (
        SIGMA
        * _STD_INFLATE
        * (_MIN_REL_STD + _ABS_JITTER_MS / short.median_ms + _CROSS_SESSION_REL)
    )
    assert got < summed, (
        "quadrature must be strictly tighter than a sum, or nothing is being tested"
    )


@matrix_exempt("pure arithmetic on measured statistics; no kernel, shape or dtype to sweep")
def test_a_ratio_valued_pin_takes_no_absolute_jitter_term():
    """`assert_host_dispatch` pins a dimensionless RATIO, so a millisecond constant cannot apply.

    ``band_for``'s ``abs_jitter_ms`` is in the same units as ``cell.median_ms``, and one caller's
    "median" is 1.447 -- a ratio of two host costs, not a duration. Dividing a millisecond jitter by
    it yields a number with no meaning. It is small rather than wildly wrong, which is exactly why
    it has to be explicit: a silently-negligible unit error is one nobody finds.
    """
    ratio = Stats(1.447, _MIN_REL_STD)
    assert band_for(ratio, abs_jitter_ms=0.0) == pytest.approx(
        1.0 + SIGMA * _MIN_REL_STD * _STD_INFLATE
    )
    assert band_for(ratio, abs_jitter_ms=0.0) < band_for(ratio), (
        "the default must still apply an absolute term, or this test proves nothing"
    )


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_a_pin_harvested_by_a_DIFFERENT_estimator_is_REFUSED():
    """THE control for the estimator guard: a mismatch must raise, not quietly compare.

    `benchmark_single(iters=20)` reads `device + C/20` -- it carries the per-call host-dispatch
    residue -- while `benchmark_extrapolated` solves that term out and reports `device`. The
    difference is small and ONE-SIDED: measured across the 13 pinned cp8 workflow cells, 11 of 13
    extrapolated readings landed BELOW their iters=20 pin, mean ratio 0.989.

    That is the dangerous shape. On a one-sided gate a systematically-low reading never fails; it
    just makes every cell look slightly improved, forever, while the gate silently loses the margin
    it was supposed to enforce. Nothing downstream can notice, which is why this refusal exists and
    why it names the re-harvest rather than merely reporting a mismatch.
    """
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01)}, estimator=pins_mod.ESTIMATOR_SINGLE)
    with pytest.raises(ValueError, match="RE-HARVEST"):
        assert_cell(
            lambda: 1.0, (1, 2), pins, "/x/t.py",
            estimator=pins_mod.ESTIMATOR_EXTRAPOLATED,
        )


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_a_matching_estimator_compares_normally():
    """The guard must not fire on the ordinary case, or every gate in the repo goes red.

    Both directions are asserted because a guard that refuses everything passes the test above
    identically to one that works.
    """
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01)}, estimator=pins_mod.ESTIMATOR_EXTRAPOLATED)
    assert_cell(lambda: 1.0, (1, 2), pins, "/x/t.py", estimator=pins_mod.ESTIMATOR_EXTRAPOLATED)
    default_pins = _FakePins({(1, 2): _FakePin(1.0, 0.01)})
    assert_cell(lambda: 1.0, (1, 2), default_pins, "/x/t.py")


@matrix_exempt("gates one measurement against one pin; no kernel, shape or dtype to sweep")
def test_a_pin_file_predating_the_estimator_field_reads_as_the_ORIGINAL_estimator():
    """An untagged file must default to `single`, never to whatever the newest estimator is.

    Eleven of the repo's twelve pin files were written before the field existed and were all
    produced by `benchmark_single`. Defaulting the other way would silently bless them as
    extrapolated and compare them against a quantity they do not hold -- the exact failure the guard
    exists to prevent, introduced by the guard itself.
    """
    assert pins_mod.DEFAULT_ESTIMATOR == pins_mod.ESTIMATOR_SINGLE


# ── budgeting the harvest's round count from the pin it already has ────────────────────────────
@matrix_exempt("pure arithmetic on a pinned median; no kernel, shape or dtype to sweep")
def test_a_cheap_cell_is_budgeted_to_exactly_what_it_does_today():
    """Nothing may change where the current count is already affordable.

    The ceiling IS `benchmark_extrapolated`'s default, so a cell under the budget must come back
    with 35 and not 34 -- otherwise landing this function silently re-times every cheap cell too,
    and the validation would have to cover all of them rather than the expensive ones it targets.
    """
    assert rounds_for(1.76, budget_s=60.0, launches_per_round=5) == 35
    assert rounds_for(0.05, budget_s=60.0, launches_per_round=5) == 35


@matrix_exempt("pure arithmetic on a pinned median; no kernel, shape or dtype to sweep")
def test_an_expensive_cell_is_cut_and_the_FLOOR_wins_over_the_budget():
    """The 245 ms cell is the one this exists for, and the floor overriding the budget is intended.

    At a 60 s budget it lands on the floor and still costs 92 s, because below 5 rounds the inner
    median stops rejecting anything. A version that honoured the budget instead would return 1 or 2
    and report a number built from a single window -- cheaper, and not a measurement.
    """
    assert rounds_for(245.4, budget_s=60.0, launches_per_round=5) == 5
    assert rounds_for(51.4, budget_s=60.0, launches_per_round=5) == 12
    # 15 samples x 5 rounds x 5 launches x 245.4 ms ~ 92 s, over the 60 s asked for. Deliberate.
    assert 15 * 5 * 5 * 245.4e-3 > 60.0


@matrix_exempt("pure arithmetic on a pinned median; no kernel, shape or dtype to sweep")
def test_an_unpinned_cell_gets_the_FULL_count_rather_than_a_guess():
    """A FIRST harvest has no pin to budget from, and must not invent one.

    Returning the ceiling is the only answer that does not reintroduce adaptivity: any other would
    require probing the cell, and a probe-derived count is per-rank, which is the desync `mode=
    "device"` is banned for. Every "no information" spelling maps to the same answer.
    """
    for absent in (0.0, -1.0, float("nan"), float("inf")):
        assert rounds_for(absent, budget_s=60.0, launches_per_round=5) == 35


@matrix_exempt("pure arithmetic on a pinned median; no kernel, shape or dtype to sweep")
def test_a_broken_cost_model_is_REFUSED_not_clamped():
    """`budget_s` in the wrong units or a zero launch count means the caller's model is wrong.

    Clamping either would return a plausible count that silently does not honour the budget, and a
    budget nobody can tell was ignored is worse than no budget.
    """
    with pytest.raises(ValueError, match="budget_s"):
        rounds_for(1.0, budget_s=0.0, launches_per_round=5)
    with pytest.raises(ValueError, match="launches_per_round"):
        rounds_for(1.0, budget_s=60.0, launches_per_round=0)


@matrix_exempt("pure arithmetic on a pinned median; no kernel, shape or dtype to sweep")
def test_every_rank_derives_the_SAME_count_from_the_same_pin_bytes(tmp_path):
    """The failure this prevents is a DESYNC, not a wrong number.

    Ranks that disagree on the round count run different numbers of collectives and hang -- which is
    exactly why `mode="device"`, whose repetition count is adaptive per rank, is banned for anything
    collective-bearing. Simulated by parsing the same pin bytes independently per "rank", because
    that is the real path: the file is on a shared filesystem and each rank reads it itself.

    Also asserts the result is an `int`. A float would be equal across ranks too, but a caller doing
    its own `int()` or `round()` on it reopens the question one layer up, and the whole point is
    that there is nothing left to get wrong.
    """
    import json

    blob = json.dumps({"cells": {"a": {"median_ms": 51.4}, "b": {"median_ms": 245.4}}})
    (tmp_path / "pins.json").write_text(blob)
    per_rank = []
    for _rank in range(16):
        cells = json.loads((tmp_path / "pins.json").read_text())["cells"]
        per_rank.append(
            tuple(
                rounds_for(c["median_ms"], budget_s=60.0, launches_per_round=5)
                for c in cells.values()
            )
        )
    assert len(set(per_rank)) == 1, f"ranks disagreed on the round count: {sorted(set(per_rank))}"
    assert all(isinstance(r, int) for r in per_rank[0]), per_rank[0]


@matrix_exempt("pure arithmetic on a pinned median; no kernel, shape or dtype to sweep")
def test_the_round_count_reaches_a_FIXED_POINT_rather_than_demanding_re_harvests_forever():
    """THE property. `assert_cell` refuses a pin whose recorded count differs from the derived one,
    so a cell that derives a different count on every harvest is a cell that can never go green.

    Asserted as CONVERGENCE rather than as a stability bar, and the bar is why: an earlier version
    asserted "a +/-x% median wobble does not change the answer", which is neither necessary (a count
    that changes once and then settles is fine) nor sufficient (it says nothing about what happens
    when it does change). It also had to be re-tuned twice as the design moved -- 1.60x failed
    correctly, 1.30x passed, and neither number was the thing being defended.

    Simulated the way the real loop runs: seed with no pin, then re-harvest whenever the derived
    count differs from the recorded one, with the observed median jittering by +/-5% each pass
    (10x the 0.5% these cells actually reproduce to). Swept across the whole median range, because
    the failure is "some median somewhere straddles a rung boundary" and a spot check cannot see it.
    """
    import pathlib

    import tests.perf.pins as pins_mod

    def _pf(median, recorded):
        cell = {"key": ["c"], "median_ms": median, "rel_std": 0.005}
        if recorded is not None:
            cell["rounds"] = recorded
        return pins_mod.PinFile(
            pathlib.Path("/x/test_benchmark_perf_x.json"),
            {"schema": 1, "kernel": "x", "measured_on": "probe", "cells": [cell]},
        )

    kw = dict(budget_s=60.0, launches_per_round=5)
    empty = pins_mod.PinFile(
        pathlib.Path("/x/test_benchmark_perf_x.json"),
        {"schema": 1, "kernel": "x", "measured_on": "probe", "cells": []},
    )
    true_m = 1.0
    while true_m < 2000.0:
        recorded, jitter, passes = None, 1.05, 0
        pins = empty
        for passes in range(1, 8):
            want = rounds_for_cell(pins, ("c",), **kw)
            if want == recorded:
                break
            # a re-harvest: measure again at the newly derived count and record it
            recorded = want
            jitter = 1.05 if jitter < 1.0 else 0.95      # alternate, the worst case for a boundary
            pins = _pf(true_m * jitter, recorded)
        else:
            raise AssertionError(
                f"median {true_m:.3f} ms never settled: still re-deriving after 7 harvests "
                f"(recorded {recorded}, wants {rounds_for_cell(pins, ('c',), **kw)})"
            )
        assert passes <= 3, (
            f"median {true_m:.3f} ms took {passes} harvests to settle on {recorded} rounds; "
            f"the design claims at most two plus the check"
        )
        true_m *= 1.07  # fine enough that every rung boundary is approached from both sides


@matrix_exempt("pure arithmetic on a pinned median; no kernel, shape or dtype to sweep")
def test_the_hysteresis_still_MOVES_when_the_cell_genuinely_changes():
    """Hysteresis that never lets go is a stuck count, which is the opposite failure.

    One rung of slack, so a recorded count survives ~1.8x of median movement and yields beyond it.
    A cell that got 5x slower is not noise, and continuing to measure it at the old count would
    reinstate exactly the fixed-launch-count defect this item removes.
    """
    import pathlib

    import tests.perf.pins as pins_mod

    def _pf(median, recorded):
        return pins_mod.PinFile(
            pathlib.Path("/x/test_benchmark_perf_x.json"),
            {"schema": 1, "kernel": "x", "measured_on": "probe", "cells": [
                {"key": ["c"], "median_ms": median, "rel_std": 0.005, "rounds": recorded}]},
        )

    kw = dict(budget_s=60.0, launches_per_round=5)
    # recorded at the ceiling, but the cell is now 20x dearer -> the count must come down
    assert rounds_for_cell(_pf(200.0, 35), ("c",), **kw) < 35
    # recorded at the floor, but the cell is now cheap -> the count must come back up
    assert rounds_for_cell(_pf(1.0, 5), ("c",), **kw) > 5


@matrix_exempt("looks a value up in a PinFile; no kernel, shape or dtype to sweep")
def test_the_round_count_is_read_through_PinFiles_REAL_lookup_api():
    """`PinFile` has no `.get`, and a probe for one would make the whole budget INERT.

    This is not hypothetical -- the first version of `rounds_for_cell` was
    `pins.get(key) if hasattr(pins, "get") else None`, which returns "unpinned" for EVERY cell and
    therefore the ceiling for every cell. Nothing raises, no test that only checks `rounds_for`
    notices, and the item silently does nothing. So the test asserts the value DISCRIMINATES, not
    merely that the call returns an int.
    """
    import pathlib

    import tests.perf.pins as pins_mod

    pf = pins_mod.PinFile(
        pathlib.Path("/x/test_benchmark_perf_x.json"),
        {"schema": 1, "kernel": "x", "measured_on": "probe", "cells": [
            {"key": ["a", 1], "median_ms": 1.76, "rel_std": 0.005},
            {"key": ["a", 2], "median_ms": 185.8, "rel_std": 0.005, "rounds": 5},
        ]},
    )
    kw = dict(budget_s=60.0, launches_per_round=5)
    assert rounds_for_cell(pf, ("a", 1), **kw) == 35, "a cheap pinned cell must keep the ceiling"
    assert rounds_for_cell(pf, ("a", 2), **kw) == 5, "an expensive pinned cell must be cut"
    assert rounds_for_cell(pf, ("a", 99), **kw) == 5, (
        "an UNPINNED cell seeds at the FLOOR: its first pass exists only to produce a median good "
        "enough to choose a rung, and seeding at the ceiling pays 7x for that estimate"
    )


@matrix_exempt("asserts a refusal on pin metadata; no kernel, shape or dtype to sweep")
def test_a_pin_harvested_at_a_DIFFERENT_round_count_is_refused():
    """Gating with fewer rounds than the pin was harvested with causes FALSE FAILURES.

    The band comes from `rel_std`, the spread ACROSS samples measured at harvest. `rounds` is the
    median WITHIN one sample. Cut `rounds` at gate time only and the band still encodes the harvest's
    noise while the samples carry more of it -- so the gate tightens rather than loosens, which is
    the opposite of the direction that makes budgeting `rounds` safe at all.

    It is also the DEFAULT sequence rather than an edge case: a first harvest has no pin so it runs
    at the ceiling, and every later run would derive a smaller count from the pin it produced.
    """
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01, rounds=35)})
    with pytest.raises(ValueError, match=r"rounds=5 but the pin was harvested with rounds=35"):
        assert_cell(lambda: 1.0, (1, 2), pins, "/x/test_benchmark_perf_x.py", rounds_used=5)
    # matching counts pass straight through
    assert_cell(lambda: 1.0, (1, 2), pins, "/x/test_benchmark_perf_x.py", rounds_used=35)


@matrix_exempt("asserts a refusal on pin metadata; no kernel, shape or dtype to sweep")
def test_a_pin_with_NO_recorded_round_count_reads_as_the_historical_default():
    """Every pin predating the field was harvested at 35, so None means 35 -- not "skip the check".

    Reading it as unverifiable would exempt exactly the cells the check exists for: the expensive
    ones, which are the only ones whose derived count differs from the ceiling.
    """
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01)})  # rounds absent
    assert_cell(lambda: 1.0, (1, 2), pins, "/x/test_benchmark_perf_x.py", rounds_used=35)
    with pytest.raises(ValueError, match=r"no recorded value; read as the historical default"):
        assert_cell(lambda: 1.0, (1, 2), pins, "/x/test_benchmark_perf_x.py", rounds_used=8)


@matrix_exempt("asserts a refusal on pin metadata; no kernel, shape or dtype to sweep")
def test_a_gate_that_passes_no_round_count_is_not_checked_at_all():
    """The eleven other perf gates have not adopted budgeted rounds and must keep working.

    Their timer's count is whatever it defaults to, identical on both sides because nothing derives
    it, so there is nothing to disagree about. Making the check unconditional would have reddened
    every one of them for metadata they have no way to supply.
    """
    pins = _FakePins({(1, 2): _FakePin(1.0, 0.01, rounds=35)})
    assert_cell(lambda: 1.0, (1, 2), pins, "/x/test_benchmark_perf_x.py")
