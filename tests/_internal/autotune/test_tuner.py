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

"""Tests for ``fold_cp_ops._internal.autotune.tuner`` -- sequencing and the decorator.

Timing is INJECTED throughout (``measure=``), so these run on a GPU but never depend on real
performance: a test that asserted "config X is fastest" would be asserting a property of the
hardware, not of the tuner. What is asserted is the tuner's own logic -- which config it selects
given known timings, when it re-measures, and which mistakes it refuses.
"""

import pytest
import torch

from fold_cp_ops._internal.autotune import AutotuneConfig, autotune
from fold_cp_ops._internal.autotune.consensus import Consensus

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _make(times, fail_on=(), **kw):
    """Build a tuned kernel whose candidate timings are dictated by `times`.

    Args:
        times: ``{tile: ms}``. The kernel records which tile it was called with, and the injected
            timer looks the value up -- so the "measurement" is deterministic.
        fail_on: Tiles for which the kernel raises, standing in for a config that cannot run.
        **kw: Forwarded to ``@autotune``.

    Returns:
        The decorated callable; ``.autotuner`` exposes the state the tests assert on.
    """
    cur = [None]

    def measure(fn, **_):
        fn()
        return times[cur[0]]

    @autotune(
        configs=[AutotuneConfig(tile=t) for t in times], measure=measure, precompile=False, **kw
    )
    def kernel(x, tile=None):
        cur[0] = tile
        if tile in fail_on:
            raise RuntimeError(f"tile {tile} cannot run")
        return x.sum()

    return kernel


@requires_cuda
def test_the_fastest_candidate_wins():
    """The base case, so a later failure means the logic changed and not the harness."""
    k = _make({32: 3.0, 64: 1.0, 128: 2.0})
    k(torch.randn(4, 8, device="cuda"))
    assert k.autotuner.best_config == AutotuneConfig(tile=64)


@requires_cuda
def test_a_failing_candidate_is_dropped_rather_than_scored():
    """It becomes ``inf`` so every rank still reaches the reduction, then is filtered out.

    The absence matters: an ``inf`` left in the results could never win, but its presence would make
    "how many candidates survived" meaningless.
    """
    k = _make({32: 1.0, 64: 2.0}, fail_on=(32,))
    k(torch.randn(4, 8, device="cuda"))
    assert k.autotuner.best_config == AutotuneConfig(tile=64)
    assert AutotuneConfig(tile=32) not in k.autotuner.last_timings


@requires_cuda
def test_every_candidate_failing_is_a_loud_error():
    """Silently returning some config would run a kernel known not to work."""
    k = _make({32: 1.0, 64: 2.0}, fail_on=(32, 64))
    with pytest.raises(RuntimeError, match=r"every candidate config .* failed"):
        k(torch.randn(4, 8, device="cuda"))


@requires_cuda
def test_the_result_is_memoized_per_request_shape():
    """A second call with the same shape must not re-measure; a new shape must."""
    calls = []
    times = {32: 2.0, 64: 1.0}

    def measure(fn, **_):
        fn()
        calls.append(1)
        return times[cur[0]]

    cur = [None]

    @autotune(configs=[AutotuneConfig(tile=t) for t in times], measure=measure, precompile=False)
    def kernel(x, tile=None):
        cur[0] = tile
        return x.sum()

    a = torch.randn(4, 8, device="cuda")
    kernel(a)
    n = len(calls)
    kernel(a)
    assert len(calls) == n, "same shape must hit the in-process memo"
    kernel(torch.randn(4, 16, device="cuda"))
    assert len(calls) > n, "a new shape must re-measure"


@requires_cuda
def test_a_padded_pitch_shares_the_tuned_result():
    """Tuning does not depend on a row pitch, so keying on it would miss on every allocation."""
    calls = []
    cur = [None]

    def measure(fn, **_):
        fn()
        calls.append(1)
        return {32: 2.0, 64: 1.0}[cur[0]]

    @autotune(configs=[AutotuneConfig(tile=t) for t in (32, 64)], measure=measure, precompile=False)
    def kernel(x, tile=None):
        cur[0] = tile
        return x.sum()

    kernel(torch.randn(4, 128, device="cuda"))
    n = len(calls)
    kernel(torch.empty(4, 136, device="cuda")[:, :128])
    assert len(calls) == n


@requires_cuda
def test_a_pinned_config_replaces_tuning_entirely():
    """``_config=`` is the deployment path: no measurement, no cache, no collective."""
    calls = []
    k = _make({32: 1.0, 64: 2.0})
    k(torch.randn(4, 8, device="cuda"))
    before = dict(k.autotuner.last_timings)
    k(torch.randn(4, 8, device="cuda"), _config=AutotuneConfig(tile=64))
    assert k.autotuner.best_config == AutotuneConfig(tile=64)
    assert k.autotuner.last_timings == before, "pinning must not re-measure"
    assert calls == []


@requires_cuda
def test_passing_a_tuned_knob_as_a_keyword_is_refused():
    """Overriding one knob would make the measured winner and the executed kernel differ."""
    k = _make({32: 1.0, 64: 2.0})
    with pytest.raises(ValueError, match=r"tuned knobs and cannot also be passed"):
        k(torch.randn(4, 8, device="cuda"), tile=32)


def test_a_knob_the_kernel_cannot_accept_is_refused_at_decoration():
    """Otherwise it surfaces at the first call as 'every candidate failed', pointing at the sweep."""
    with pytest.raises(TypeError, match=r"declares knob\(s\) \['nosuch'\]"):

        @autotune(configs=[AutotuneConfig(nosuch=1), AutotuneConfig(nosuch=2)], precompile=False)
        def kernel(x):
            return x


def test_a_kernel_taking_kwargs_accepts_any_knob():
    """The signature check must not fire on a kernel that legitimately accepts anything."""

    @autotune(configs=[AutotuneConfig(anything=1), AutotuneConfig(anything=2)], precompile=False)
    def kernel(x, **kw):
        return x

    assert kernel.autotuner is not None


@requires_cuda
def test_autotune_disabled_takes_the_first_admissible_config():
    """A correctness run and a pinned perf gate must measure a kernel, never a sweep."""
    k = _make({32: 5.0, 64: 1.0})
    import os

    os.environ["CPO_AUTOTUNE"] = "0"
    try:
        k(torch.randn(4, 8, device="cuda"))
        assert k.autotuner.best_config == AutotuneConfig(tile=32), "declaration order, not fastest"
    finally:
        os.environ.pop("CPO_AUTOTUNE", None)


def test_the_tie_break_is_deterministic_not_dict_order():
    """After a cross-rank reduction exact ties are common, and this is the last place ranks diverge."""
    tied = {AutotuneConfig(tile=64): 1.0, AutotuneConfig(tile=32): 1.0}
    reversed_ = {AutotuneConfig(tile=32): 1.0, AutotuneConfig(tile=64): 1.0}
    assert Consensus.pick(tied) == Consensus.pick(reversed_)


def test_consensus_is_inert_in_a_single_process():
    """Every local kernel takes this path; it must add no collectives and no divergence report."""
    c = Consensus()
    assert not c.enabled and c.world_size == 1 and c.rank == 0
    assert c.divergent_failures() == ()
    assert c.agree_timings({AutotuneConfig(tile=1): 2.0}) == {AutotuneConfig(tile=1): 2.0}
    assert c.agree_timings({AutotuneConfig(tile=1): float("inf")}) == {}


@requires_cuda
def test_the_warm_path_does_not_re_run_validity_over_the_pool():
    """A repeat call at a known shape is one key build and one dict hit -- nothing O(pool).

    **This is a performance contract with a correctness consequence, which is why it is asserted
    rather than left to the perf gate.** The tuned entry is a production path; anything the warm
    dispatch does per call is added to every launch. It used to rebuild the tuned-knob name set (a
    walk over every config, copying each one's kwargs) AND re-run `validity` over the whole pool on
    every call. Measured on an H100, that was ~20 us per launch, which roughly DOUBLED a
    launch-bound GEMM -- so a tuned kernel could be slower than the fixed one it was tuning, and the
    perf gate would report a kernel-selection regression that was really a dispatch regression.

    Counting `validity` calls is the mechanical way to state it: once per config during the sweep,
    and never again for the same shape.
    """
    calls = {"n": 0}

    def counting_validity(config, request):
        """Admit everything, and count how many times the tuner asks."""
        calls["n"] += 1
        return True

    k = _make({64: 3.0, 128: 1.0}, validity=counting_validity)
    x = torch.zeros(8, device="cuda")
    k(x)
    after_tune = calls["n"]
    assert after_tune >= 2, "validity must be consulted once per config while tuning"
    for _ in range(5):
        k(x)
    assert calls["n"] == after_tune, (
        f"validity was called {calls['n'] - after_tune} more times across 5 warm calls; the warm "
        f"path must not re-run it over the pool"
    )


@requires_cuda
def test_a_pinned_config_skips_binding_and_the_key_entirely():
    """`_config=` does no argument binding, no key build and no validity -- it just calls.

    That is what makes it the deployment path: tuning is a development-time activity, and the
    per-call cost of re-deriving "which shape is this" is pure overhead once the answer is known.
    """
    calls = {"n": 0}

    def counting_validity(config, request):
        """Admit everything, and count how many times the tuner asks."""
        calls["n"] += 1
        return True

    k = _make({64: 3.0, 128: 1.0}, validity=counting_validity)
    x = torch.zeros(8, device="cuda")
    k(x, _config=AutotuneConfig(tile=64))
    assert calls["n"] == 0, "a pinned config must not consult validity at all"
    assert k.autotuner.best_config == AutotuneConfig(tile=64)


def test_the_fast_bind_agrees_with_inspect_for_an_ordinary_signature():
    """The `zip`-based binding must produce exactly what `Signature.bind_partial` would.

    The fast path exists because `bind_partial` costs several microseconds on every dispatch. It is
    only correct for a signature with no ``*args``/``**kwargs``, and the class checks that -- but a
    silent disagreement between the two would mean the request key was built from differently-named
    arguments, so two different shapes could share a tuned result. This compares them directly.
    """
    import inspect

    from fold_cp_ops._internal.autotune.space import ConfigSpace
    from fold_cp_ops._internal.autotune.tuner import Autotuner

    def kernel(a, b, c=3, *, d=4, tile=None):
        """A plain signature: positional, defaulted and keyword-only, no var-args."""
        return a

    tuner = Autotuner(kernel, space=ConfigSpace([AutotuneConfig(tile=1)]), precompile=False)
    assert tuner._simple_signature
    for args, kwargs in (((1, 2), {}), ((1, 2, 3), {"d": 9}), ((1,), {"b": 2}), ((), {"a": 1})):
        fast = tuner._bind(args, kwargs)
        slow = dict(inspect.signature(kernel).bind_partial(*args, **kwargs).arguments)
        assert fast == slow, f"fast bind disagreed for {args} {kwargs}: {fast} != {slow}"


def test_a_var_args_signature_falls_back_to_inspect():
    """A ``**kwargs`` signature cannot be bound positionally, so the fast path must not claim it.

    `test_a_kernel_taking_kwargs_accepts_any_knob` covers the behaviour; this covers the mechanism,
    because a fast path that silently mis-bound such a signature would corrupt the request key
    rather than raise.
    """
    from fold_cp_ops._internal.autotune.space import ConfigSpace
    from fold_cp_ops._internal.autotune.tuner import Autotuner

    def kernel(x, **kwargs):
        """A signature the positional zip cannot handle."""
        return x

    tuner = Autotuner(kernel, space=ConfigSpace([AutotuneConfig(tile=1)]), precompile=False)
    assert not tuner._simple_signature, (
        "a **kwargs signature must take the inspect path; the zip would drop the extras"
    )


def test_a_gate_must_be_keyword_only():
    """A positionally-passable gate is refused at DECORATION time, naming the fix.

    The dispatch check is ``kwargs.get(gate)``. A gate that can arrive positionally is invisible to
    it, so ``kernel(x, True)`` would tune while the call site reads as asking it not to -- the exact
    inversion the flag exists to prevent, and silent.
    """
    with pytest.raises(TypeError, match=r"must be KEYWORD-ONLY"):

        @autotune(configs=[AutotuneConfig(tile=1)], gate="do_autotune", precompile=False)
        def positional_gate(x, tile=None, do_autotune=False):
            """A gate the caller could pass positionally."""
            return x

    with pytest.raises(TypeError, match=r"has no such parameter"):

        @autotune(configs=[AutotuneConfig(tile=1)], gate="nope", precompile=False)
        def absent_gate(x, *, tile=None):
            """No gate parameter at all."""
            return x


@requires_cuda
def test_the_gate_off_goes_straight_to_the_kernel():
    """With the gate falsy nothing is measured, nothing is bound, and validity is never consulted.

    **This is what lets ONE function be both entry points.** Three names for one kernel -- fixed,
    decorated, and a forwarder -- is three places for a caller to reach the wrong one, and the
    forwarder in particular had no reason to exist. Collapsing them is only safe if the fixed path
    stays a fixed path, which is what this asserts: the caller's own knobs run, untouched.
    """
    calls = {"validity": 0}

    def counting_validity(config, request):
        """Admit everything, and count how many times the tuner asks."""
        calls["validity"] += 1
        return True

    seen = []

    @autotune(
        configs=[AutotuneConfig(tile=64), AutotuneConfig(tile=128)],
        validity=counting_validity,
        gate="do_autotune",
        precompile=False,
        measure=lambda fn, **_: (fn(), 1.0)[1],
    )
    def kernel(x, tile=None, *, do_autotune=False):
        """Records the tile it was called with."""
        seen.append(tile)
        return x.sum()

    x = torch.zeros(8, device="cuda")
    kernel(x, 256)  # the caller's own config, positionally
    assert seen == [256], "the gate-off path must pass the caller's knob through untouched"
    assert calls["validity"] == 0, "the gate-off path must not consult validity at all"
    assert kernel.autotuner.best_config is None, "nothing was chosen, so nothing may be recorded"

    kernel(x, do_autotune=True)
    assert kernel.autotuner.best_config is not None
    assert calls["validity"] >= 2, "the gate-on path must run validity over the pool"


@requires_cuda
def test_a_knob_passed_positionally_alongside_the_gate_is_refused():
    """The conflict check reads the BOUND request, so a positional knob is caught too.

    For a kernel whose tile shape is its sixth argument, positional is how callers actually pass it.
    A check against keywords alone let it through and it died several frames later as
    ``TypeError: got multiple values for argument 'tile'`` -- which names the symptom, not the
    mistake, and points at the tuner rather than at the call.
    """

    @autotune(
        configs=[AutotuneConfig(tile=64), AutotuneConfig(tile=128)],
        gate="do_autotune",
        precompile=False,
        measure=lambda fn, **_: (fn(), 1.0)[1],
    )
    def kernel(x, tile=None, *, do_autotune=False):
        """A kernel whose tuned knob is positionally passable."""
        return x.sum()

    x = torch.zeros(8, device="cuda")
    with pytest.raises(ValueError, match=r"tuned knobs and cannot also be passed"):
        kernel(x, 256, do_autotune=True)
