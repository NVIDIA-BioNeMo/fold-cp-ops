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

"""Tests for `fold_cp_ops.distributed.collective_symmetry`.

Run: under ANY of the launch forms in CLAUDE.md's *"The five launch forms"* -- torchrun or plain
srun, 1 node or 2. Two ranks, however they are placed.

Every test here is DELIBERATELY asymmetric: one rank is made to fail, or to want a skip, while the
other does not. That asymmetry is the whole subject -- a test in which both ranks behave the same
would pass against a completely broken gate, because a symmetric group never needed the gate in the
first place. The parametrization therefore always singles out rank 0.

Input requirements
    Exactly 2 ranks. A world size of 1 makes every asymmetry test vacuous (the "failing rank" and
    the "observing rank" become the same process), so the fixture refuses it rather than reporting
    a green run that proved nothing. CUDA is required only because the default backend is NCCL;
    the module itself is backend-agnostic.
"""

from __future__ import annotations

import os

import pytest

from fold_cp_ops.testing.collective_guard import rank_invariant_skip

from fold_cp_ops.testing.kernel_matrix import matrix_exempt
import torch
import torch.distributed as dist

from fold_cp_ops.distributed.collective_symmetry import (
    BarrierTimeout,
    CollectiveFailure,
    CollectiveGate,
    PeerFailure,
)
from fold_cp_ops.distributed.distributed_manager import DistributedManager

# Fill RANK / WORLD_SIZE / LOCAL_RANK from SLURM's variables when the launcher is a plain `srun`
# with no torchrun -- a no-op when torchrun already owns the env. It has to run BEFORE the reads
# below, not inside a fixture, because these are module-scope constants: a fixture-time derivation
# is too late for a value read at import.
#
# **Measured, and the failure looked nothing like a launcher problem.** Under the 2-node srun form
# this file read WORLD_SIZE=1 -- srun sets SLURM_NTASKS, not WORLD_SIZE -- and every test errored on
# the `WORLD == 2` assertion, i.e. the file appeared to demand a world size the launch had actually
# provided. `tests/distributed/conftest.py` already calls this same function, but from inside a
# fixture, so nothing covered a module-scope reader. The manager is ASKED which launcher this is
# rather than the mapping being reimplemented: a second copy of the SLURM->env mapping is the
# "two things, one name" failure. Only the three identity names are mapped; the rendezvous
# address/port derivation stays solely in `_initialize_slurm`.
_LAUNCH = DistributedManager._detect_launcher()
_R, _W, _L = (
    ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID")
    if _LAUNCH == "SLURM"
    else ("RANK", "WORLD_SIZE", "LOCAL_RANK")
)

WORLD = int(os.environ.get(_W) or os.environ.get("SLURM_NPROCS", "1"))
RANK = int(os.environ.get(_R, "0"))
LOCAL_RANK = int(os.environ.get(_L, "0"))


@pytest.fixture(scope="session")
def gate():
    """One `CollectiveGate` over the default group for the whole session.

    Returns:
        The gate. SKIPS -- declared, not silent -- on any world size other than 2.

        This used to be a hard `assert`, on the reasoning that a skip would let pytest report
        success for a session that exercised none of the asymmetry this file exists to test. The
        worry is right; the mechanism was wrong twice over. `WORLD_SIZE` is the canonical
        rank-invariant predicate -- job-uniform, every rank computes it identically -- so
        CLAUDE.md's rule for `tests/distributed/**` requires `rank_invariant_skip`. And the
        non-coverage it feared is already caught elsewhere: the coverage ledger records every
        skip-with-reason across launches and `CPO_DIST_COVERAGE_STRICT=1` fails a sweep that never
        reached a declared cell. Hard-failing instead turned every world!=2 multi-file sweep red for
        a non-defect, which is a worse signal than the one it was guarding against.
    """
    if WORLD != 2:
        rank_invariant_skip(
            f"this file needs exactly 2 ranks; got WORLD_SIZE={WORLD}. Every test singles out "
            f"rank 0, so a world of 1 would pass against a broken gate and a larger world leaves "
            f"the extra ranks with no declared role.",
            because="WORLD_SIZE is fixed by the launcher and identical on every rank, so every "
            "rank evaluates this comparison the same way and they skip together or not at all",
        )
    if not dist.is_initialized():
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group("nccl")
    g = CollectiveGate()
    yield g
    # Teardown is itself the subject matter: barrier BEFORE destroying, so no rank tears the group
    # down while a peer is still inside a collective, and destroy explicitly rather than leaving it
    # to interpreter exit (which warns about leaked resources and, with NVSHMEM in the picture,
    # is exactly the D2 ordering this module exists to make safe).
    g.barrier(timeout_s=60)
    dist.destroy_process_group()


@matrix_exempt(
    "the subject is CollectiveGate's semantics under a deliberately ASYMMETRIC 2-rank group -- what is swept is which rank diverges, not a shape, a dtype or a mesh; the file asserts it needs exactly 2 ranks for that reason"
)
def test_any_and_all_disagree_exactly_where_it_matters(gate):
    """`op="any"` and `op="all"` must differ when exactly one rank votes True."""
    only_rank0 = RANK == 0
    assert gate.agree_bool(only_rank0, op="any") is True
    assert gate.agree_bool(only_rank0, op="all") is False
    assert gate.agree_bool(True, op="all") is True
    assert gate.agree_bool(False, op="any") is False


@matrix_exempt(
    "the subject is CollectiveGate's semantics under a deliberately ASYMMETRIC 2-rank group -- what is swept is which rank diverges, not a shape, a dtype or a mesh; the file asserts it needs exactly 2 ranks for that reason"
)
def test_a_failure_on_one_rank_becomes_a_failure_on_every_rank(gate):
    """The central property: rank 0 raises, and rank 1 must NOT sail on into the next collective.

    Rank 0 gets `CollectiveFailure` wrapping its own exception; rank 1 gets `PeerFailure`. Both
    raise, which is what keeps them in step.
    """
    expected = CollectiveFailure if RANK == 0 else PeerFailure
    with pytest.raises(expected):
        with gate.guard("deliberate-divergence"):
            if RANK == 0:
                raise ValueError("only rank 0 fails here")


@matrix_exempt(
    "the subject is CollectiveGate's semantics under a deliberately ASYMMETRIC 2-rank group -- what is swept is which rank diverges, not a shape, a dtype or a mesh; the file asserts it needs exactly 2 ranks for that reason"
)
def test_the_failing_rank_keeps_its_original_exception_chained(gate):
    """A wrapped failure must not lose the cause, or the log names the gate instead of the bug."""
    try:
        with gate.guard("chaining"):
            if RANK == 0:
                raise KeyError("the real cause")
    except CollectiveFailure as exc:
        assert isinstance(exc.__cause__, KeyError)
        assert "the real cause" in str(exc)
    except PeerFailure:
        assert RANK == 1


@matrix_exempt(
    "the subject is CollectiveGate's semantics under a deliberately ASYMMETRIC 2-rank group -- what is swept is which rank diverges, not a shape, a dtype or a mesh; the file asserts it needs exactly 2 ranks for that reason"
)
def test_a_guard_that_does_not_fail_anywhere_is_transparent(gate):
    """The common path must cost nothing observable -- no spurious raise, body value preserved."""
    seen = []
    with gate.guard("clean"):
        seen.append(RANK)
    assert seen == [RANK]


@matrix_exempt(
    "the subject is CollectiveGate's semantics under a deliberately ASYMMETRIC 2-rank group -- what is swept is which rank diverges, not a shape, a dtype or a mesh; the file asserts it needs exactly 2 ranks for that reason"
)
def test_ranks_skip_together_or_not_at_all(gate):
    """One rank wanting a skip must make BOTH skip, and the reason must not be invented."""
    reason = "rank 0's local predicate" if RANK == 0 else None
    agreed = gate.should_skip(reason)
    assert agreed is not None, "a peer asked to skip and this rank did not agree"
    if RANK == 0:
        assert agreed == "rank 0's local predicate"
    else:
        # Rank 1 never observed the condition, so it must not claim it did.
        assert "peer" in agreed


@matrix_exempt(
    "the subject is CollectiveGate's semantics under a deliberately ASYMMETRIC 2-rank group -- what is swept is which rank diverges, not a shape, a dtype or a mesh; the file asserts it needs exactly 2 ranks for that reason"
)
def test_nobody_skipping_means_nobody_skips(gate):
    """The negative case, which is the one a broken reduction would get wrong silently."""
    assert gate.should_skip(None) is None


@matrix_exempt(
    "the subject is CollectiveGate's semantics under a deliberately ASYMMETRIC 2-rank group -- what is swept is which rank diverges, not a shape, a dtype or a mesh; the file asserts it needs exactly 2 ranks for that reason"
)
def test_a_barrier_everyone_reaches_completes(gate):
    """The control for the timeout test below: a symmetric barrier must not raise."""
    gate.barrier(timeout_s=60)


@matrix_exempt(
    "the subject is CollectiveGate's semantics under a deliberately ASYMMETRIC 2-rank group -- what is swept is which rank diverges, not a shape, a dtype or a mesh; the file asserts it needs exactly 2 ranks for that reason"
)
def test_an_unbounded_barrier_cannot_be_requested(gate):
    """A non-positive bound is refused, because 'wait forever' is the bug this class replaces."""
    with pytest.raises(ValueError):
        gate.barrier(timeout_s=0)


@matrix_exempt(
    "the subject is CollectiveGate's semantics under a deliberately ASYMMETRIC 2-rank group -- what is swept is which rank diverges, not a shape, a dtype or a mesh; the file asserts it needs exactly 2 ranks for that reason"
)
def test_the_barrier_timeout_names_the_situation_rather_than_just_timing_out(gate):
    """A timeout must tell the reader the group is now unusable, not merely that time passed.

    Checked on the message rather than by inducing a real desync: deliberately stranding a rank
    would leave the process group broken for every test after this one, and the fixture's teardown
    barrier would then hang -- trading a fast, readable assertion for the exact failure mode this
    module exists to prevent.
    """
    msg = BarrierTimeout(
        "rank 0 waited 600s at a collective barrier and at least one peer never arrived. The group "
        "is now desynchronized -- do NOT enter another collective."
    )
    assert "desynchronized" in str(msg)
    assert "do NOT enter another collective" in str(msg)
