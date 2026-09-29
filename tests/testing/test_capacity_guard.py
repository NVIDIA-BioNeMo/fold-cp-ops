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

"""Unit tests for :mod:`fold_cp_ops.testing.capacity_guard`.

GPU-free and group-free: ``gated_skip`` is monkeypatched to a recorder, so these assert the guard's
DECISION LOGIC -- which exceptions count as capacity, and whether the collective is reached on the
success path -- without needing a process group. The collective behaviour of ``gated_skip`` itself is
covered by ``tests/testing/test_collective_guard.py``; duplicating it here would test that module
twice and this one not at all.
"""

import pytest
import torch

from fold_cp_ops.testing import capacity_guard
from fold_cp_ops.testing.capacity_guard import HEAP_EXHAUSTED, capacity_gate, is_capacity_error
from fold_cp_ops.testing.numeric_guard import numeric_exempt

pytestmark = numeric_exempt("asserts guard control flow, not a computed tensor")


class _Recorder:
    """Stand-in for ``gated_skip`` that records every call instead of reducing or skipping.

    Input requirements:
        None. Call it exactly as ``gated_skip(local_reason)`` is called.

    Returns:
        None from every call; ``reasons`` accumulates one entry per call, so a test can assert BOTH
        that the collective was reached and what reason this rank contributed.
    """

    def __init__(self):
        self.reasons = []

    def __call__(self, local_reason=None, **_kwargs):
        self.reasons.append(local_reason)


@pytest.fixture
def recorder(monkeypatch):
    """Replace ``capacity_gate``'s ``gated_skip`` with a :class:`_Recorder`.

    Patches the name in the capacity_guard MODULE, not in collective_guard: the guard imported the
    function at module load, so rebinding the source would not be seen by the caller under test.

    Returns:
        The :class:`_Recorder` instance the guard will call.
    """
    rec = _Recorder()
    monkeypatch.setattr(capacity_guard, "gated_skip", rec)
    return rec


def test_torch_oom_is_a_capacity_error():
    """The caching allocator's own OOM type counts."""
    assert is_capacity_error(torch.cuda.OutOfMemoryError("CUDA out of memory"))


@pytest.mark.parametrize("fragment", HEAP_EXHAUSTED)
def test_every_declared_heap_message_is_a_capacity_error(fragment):
    """Each declared symmetric-heap fragment is recognised inside a RuntimeError.

    Parametrized over the constant rather than restating the strings, so a fragment added to
    :data:`HEAP_EXHAUSTED` without a matching behaviour cannot pass silently.
    """
    assert is_capacity_error(RuntimeError(f"something went wrong: {fragment}"))


def test_a_bare_runtime_error_is_not_a_capacity_error():
    """A RuntimeError with no heap message is a DEFECT and must propagate.

    This is the whole reason the predicate matches on the MESSAGE: a bare ``except RuntimeError``
    around a build swallows every real failure the gate exists to surface.
    """
    assert not is_capacity_error(RuntimeError("kernel produced the wrong answer"))
    assert not is_capacity_error(ValueError("nvshmem_malloc failed"))  # right text, wrong type


def test_the_collective_is_reached_on_the_SUCCESS_path(recorder):
    """A step that does not raise still reaches ``gated_skip``, with ``None``.

    This is the property the guard exists for and the one every hand-rolled version got wrong:
    calling ``gated_skip`` only from an ``except`` block means only the RAISING ranks enter the
    all_reduce, so they block on peers who never arrive.
    """
    assert capacity_gate("cheap step", lambda: "value") == "value"
    assert recorder.reasons == [None], "success path did not reach the collective exactly once"


def test_a_capacity_failure_is_captured_and_reported_not_raised(recorder):
    """An OOM becomes a reason handed to the collective, not an exception out of the gate."""

    def _boom():
        raise torch.cuda.OutOfMemoryError("Tried to allocate 5.96 GiB")

    assert capacity_gate("recv (symmetric)", _boom) is None
    assert len(recorder.reasons) == 1
    reason = recorder.reasons[0]
    assert reason is not None and "recv (symmetric)" in reason and "5.96 GiB" in reason


def test_a_non_capacity_exception_propagates_and_skips_the_collective(recorder):
    """A defect must fail the cell, and must NOT enter the collective.

    Entering it would be worse than the raise: the failing rank would reduce while its peers, who did
    not fail, reduce too -- turning one rank's real bug into a group-wide skip that hides it.
    """

    def _boom():
        raise RuntimeError("epilogue produced NaN")

    with pytest.raises(RuntimeError, match="epilogue produced NaN"):
        capacity_gate("build", _boom)
    assert recorder.reasons == [], "a defect reached the collective; it must propagate instead"
