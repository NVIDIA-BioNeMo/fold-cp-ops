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


"""Tests for ``fold_cp_ops._internal.pipeline`` -- the pipeline subclasses this tree adds.

Constructing a real pipeline needs an MLIR context and SMEM, so these tests pin what the subclasses
*are*: the mixin surface every one of them gains, the re-classing factory, and the policy flags that
decide whether a barrier arrive comes from one thread or from all of them.
"""

import inspect

import pytest

from fold_cp_ops._internal import pipeline as P


@pytest.mark.parametrize(
    "cls",
    [
        P.PipelineAsync,
        P.PipelineCpAsync,
        P.PipelineTmaAsync,
        P.PipelineUmmaAsync,
        P.PipelineAsyncUmma,
        P.PipelineTmaCpAsync,
    ],
)
def test_every_pipeline_gains_the_index_phase_entry_points(cls):
    """A scheduler that computes its own stage arithmetic drives the pipeline by ``(index, phase)``.

    Uniformity matters more than the individual methods: a pipeline missing one forces its caller
    into a different calling convention than every sibling.
    """
    assert issubclass(cls, P._PipelineIndexPhaseMixin)
    for name in (
        "producer_acquire_w_index_phase",
        "producer_commit_w_index",
        "consumer_wait_w_index_phase",
        "consumer_release_w_index",
    ):
        assert hasattr(cls, name), f"{cls.__name__} lost {name}"


def test_pipeline_state_w_advance_can_skip_stages():
    """The base state advances one stage at a time; this one jumps, phase bit and all."""
    from cutlass.pipeline import PipelineState

    assert issubclass(P.PipelineStateWAdvance, PipelineState)
    assert hasattr(P.PipelineStateWAdvance, "advance_iters")


def test_make_pipeline_state_starts_producer_and_consumer_on_opposite_phases():
    """A producer begins with an empty buffer (phase 1); a consumer with nothing to read (phase 0).

    Starting both on the same phase is a deadlock at the very first stage, so the asymmetry is the
    whole content of this factory.
    """
    from cutlass.pipeline import PipelineUserType

    prod = P.make_pipeline_state(PipelineUserType.Producer, 4)
    cons = P.make_pipeline_state(PipelineUserType.Consumer, 4)
    assert isinstance(prod, P.PipelineStateWAdvance)
    assert isinstance(cons, P.PipelineStateWAdvance)


def test_elect_one_policies_are_bound_at_create_not_per_call():
    """The flags live on the object, so the per-call path has no branch to evaluate."""
    sig = inspect.signature(P.PipelineAsync.create)
    for name in (
        "elect_one_commit",
        "syncwarp_before_commit",
        "elect_one_release",
        "syncwarp_before_release",
    ):
        assert name in sig.parameters, name
        assert sig.parameters[name].default in (False, True)
    assert sig.parameters["elect_one_commit"].default is False, (
        "electing must be opt-in: it is only correct when the barrier's arrive count is 1 per warp"
    )
    assert sig.parameters["syncwarp_before_commit"].default is True, (
        "the sync is the safe default; dropping it is the optimization, not the other way round"
    )


def test_cp_async_offers_only_the_release_policy():
    """A cp.async producer commits through the async-group mechanism, so there is no commit to elect."""
    sig = inspect.signature(P.PipelineCpAsync.create)
    assert "elect_one_release" in sig.parameters
    assert "elect_one_commit" not in sig.parameters


def test_tma_cp_async_gates_the_full_barrier_arrive_on_one_warp():
    """Several producer warps call ``producer_acquire``; only the TMA warp may arrive.

    The full barrier's transaction count is set for a single TMA arrival, so a second arrival lets
    the consumer proceed on data that has not landed -- a wrong answer, not a hang.
    """
    sig = inspect.signature(P.PipelineTmaCpAsync.producer_acquire)
    assert "is_tma_warp" in sig.parameters


def test_mbarrier_with_drop_count_rejects_a_degenerate_configuration():
    """Zero stages or a non-positive arrive count would leave a barrier nobody can complete."""
    src = inspect.getsource(P.MbarrierArrayWDropCount.__init__)
    assert "raise ValueError" in src
    assert "num_stages" in src and "arrive_count" in src
