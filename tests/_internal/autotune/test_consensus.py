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

"""Tests for ``fold_cp_ops._internal.autotune.consensus`` -- how ranks agree on one winner.

**Autotuning is SPMD.** Every rank is handed the same candidate list, runs the same sweep in the
same order, and must end up choosing the same config. Nothing about that is checked by the kernel:
if two ranks chose differently, a kernel containing a collective deadlocks minutes later with
nothing pointing back here.

Three properties carry that, and each has its own failure mode:

* the timings are max-reduced, so every rank holds the SLOWEST rank's number -- a per-rank number
  would make the winner "whichever rank waited least";
* a config that failed anywhere is dropped everywhere, via the ``inf`` contract;
* the tie-break is on the canonical key, not on dict order, because after a reduction exact ties are
  common and ``min(timings, key=timings.get)`` returns whichever key the dict yields first.

The multi-rank half runs under torchrun with a gloo group -- no GPU needed, since what is being
tested is the agreement logic and not the kernels. Single-process, `Consensus` is inert by design,
and the tests below assert that too: a local sweep must not pay for machinery it does not need.

Run the distributed half with::

    CPO_CACHE_ENABLED=0 torchrun --nproc_per_node=2 -m pytest -q \\
        tests/_internal/autotune/test_consensus.py -k distributed
"""

import math
import os

import pytest

from fold_cp_ops._internal.autotune import AutotuneConfig
from fold_cp_ops._internal.autotune.consensus import Consensus

_A, _B, _C = AutotuneConfig(tile=64), AutotuneConfig(tile=128), AutotuneConfig(tile=256)


def _under_torchrun() -> bool:
    """True only for a genuine multi-rank torchrun launch."""
    return (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


# ── single process: the machinery must be inert ────────────────────────────────────────────────
def test_consensus_is_inert_without_a_process_group():
    """No group -> world size 1, rank 0, and no reduction. A local sweep pays nothing."""
    c = Consensus()
    assert c.enabled is False
    assert c.world_size == 1 and c.rank == 0
    c.barrier()  # must not raise


def test_a_failed_config_is_dropped_even_single_process():
    """The ``inf`` contract holds with one rank too, so the local and distributed paths agree.

    If ``inf`` only meant "failed" under a group, a local sweep would carry a failed config into
    `pick` and choose it whenever every other config was slower than infinity -- which is never, but
    the asymmetry would be a trap for the next person to read `_sweep`.
    """
    got = Consensus().agree_timings({_A: 1.0, _B: float("inf")})
    assert got == {_A: 1.0}


def test_pick_prefers_the_fastest():
    """The base case, so a later failure means the tie-break changed and not the comparison."""
    assert Consensus.pick({_A: 2.0, _B: 1.0, _C: 3.0}) == _B


def test_pick_breaks_exact_ties_on_the_key_not_on_dict_order():
    """Two dicts with the same values in different insertion orders must choose the same config.

    This is the LAST place ranks can diverge: after `agree_timings` every rank holds identical
    values, and exact ties are common once timings are quantized by a reduction.
    """
    forward = Consensus.pick({_A: 1.0, _B: 1.0, _C: 1.0})
    backward = Consensus.pick({_C: 1.0, _B: 1.0, _A: 1.0})
    assert forward == backward


def test_pick_refuses_an_empty_or_all_failed_sweep():
    """Every candidate failing is a loud error, never an arbitrary pick.

    An arbitrary pick here would run a config that is KNOWN not to work, which is a crash at best
    and a hang under a collective. The message names the verbose flag, because the per-config errors
    are the only way to find out why.
    """
    with pytest.raises(ValueError, match=r"no timings"):
        Consensus.pick({})
    with pytest.raises(ValueError, match=r"every candidate config failed"):
        Consensus.pick({_A: float("inf"), _B: float("inf")})


def test_a_nan_timing_is_treated_as_a_failure():
    """NaN is not a fast config. It compares false against everything, so it must be filtered.

    Without the filter, ``min`` over a set containing NaN returns a value that depends on iteration
    order -- the same class of non-determinism the tie-break exists to remove.
    """
    assert Consensus.pick({_A: float("nan"), _B: 2.0}) == _B
    with pytest.raises(ValueError, match=r"every candidate config failed"):
        Consensus.pick({_A: float("nan")})


def test_is_distributed_env_is_about_the_LAUNCHER_not_the_group(monkeypatch):
    """The gap between "launched distributed" and "group initialized" is a real failure mode.

    A torchrun job that autotunes before ``init_process_group`` would tune per-rank with no
    consensus and no warning. The tuner refuses that, and this is the predicate it refuses on -- so
    it must answer for the LAUNCHER even when no group exists.
    """
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert Consensus.is_distributed_env() is False
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert Consensus.is_distributed_env() is True
    assert Consensus().enabled is False, "the env alone must not make consensus think it is enabled"
    # WORLD_SIZE=1 is NOT a distributed env, and this case is why the clause exists:
    # `DistributedManager._setup` exports RANK/WORLD_SIZE after a SLURM parse, so
    # any `srun --ntasks=1` -- including the documented way to run this suite on a cluster -- sets
    # RANK=0/WORLD_SIZE=1 process-globally. Refusing there is vacuous: one rank cannot disagree with
    # itself and no collective can deadlock against a peer that does not exist. Measured: without
    # this, 84 tests fail under `srun ... pytest tests/ -n 8`.
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert Consensus.is_distributed_env() is False, "one rank is not a distributed env"
    # Unparseable -> fail toward the refusal, since a loud error beats per-rank tuning on a real job.
    monkeypatch.setenv("WORLD_SIZE", "not-a-number")
    assert Consensus.is_distributed_env() is True


# ── multi rank: the agreement itself, over gloo ────────────────────────────────────────────────
@pytest.fixture(scope="module")
def gloo_group():
    """A CPU-only process group from the torchrun environment; skips when single-process.

    Gloo rather than NCCL deliberately: what is under test is the agreement logic, which needs no
    GPU. Requiring one would make this the kind of test that only runs on a machine that already
    has the thing it is meant to protect.

    Yields:
        The world size.
    """
    if not _under_torchrun():
        pytest.skip("needs torchrun with WORLD_SIZE>1")
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    yield dist.get_world_size()
    dist.barrier()
    dist.destroy_process_group()


def test_distributed_timings_reduce_to_the_slowest_rank(gloo_group):
    """Every rank ends with the SLOWEST rank's number for each config.

    A per-rank number is that rank's own view of a collective it participated in, so the config that
    "wins" would be whichever rank happened to wait least -- an artifact of skew, not a property of
    the kernel.
    """
    import torch.distributed as dist

    rank = dist.get_rank()
    # Rank r reports r+1 ms for _A and a constant for _B, so the max is world_size for _A.
    got = Consensus().agree_timings({_A: float(rank + 1), _B: 5.0})
    assert got[_A] == float(gloo_group)
    assert got[_B] == 5.0


def test_distributed_a_config_failing_anywhere_is_dropped_everywhere(gloo_group):
    """The ``inf`` contract: one rank's failure removes the config from every rank's pool.

    The reduction happens unconditionally, OUTSIDE the try that catches a candidate's exception --
    which is what makes this work. A rank that skipped the reduce because it caught an error would
    leave its peers waiting inside a collective forever.
    """
    import torch.distributed as dist

    failed_on_rank0 = float("inf") if dist.get_rank() == 0 else 1.0
    got = Consensus().agree_timings({_A: failed_on_rank0, _B: 2.0})
    assert _A not in got, "a config that failed on ANY rank must not survive on any other"
    assert got == {_B: 2.0}


def test_distributed_divergence_is_reported_not_absorbed(gloo_group):
    """A config that failed on some ranks and not others is NAMED, via the extra MIN reduce.

    Dropping it silently would be defensible -- the config is unusable either way -- but it hides a
    setup bug: identical ranks running identical code should fail identically, so divergence means
    the ranks are not identical (a device difference, a stale artifact, a non-pure validity rule).
    """
    import torch.distributed as dist

    if dist.get_world_size() < 2:
        pytest.skip("divergence needs at least two ranks")
    c = Consensus()
    c.agree_timings({_A: float("inf") if dist.get_rank() == 0 else 1.0, _B: 2.0})
    assert _A in c.divergent_failures(), (
        "a config that failed on one rank and succeeded on another must be reported as divergent"
    )


def test_distributed_every_rank_picks_the_same_config(gloo_group):
    """The end-to-end property: reduce, then pick, and all ranks agree without further messages.

    `pick` needs no communication precisely because everything before it produced something
    identical on every rank. That is what makes it correct even if a rank reached it early.
    """
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    # Deliberately asymmetric local timings; after the max-reduce they must agree.
    local = {_A: 1.0 + rank * 0.5, _B: 1.2, _C: 1.0 + (1 - rank % 2) * 0.5}
    winner = Consensus.pick(Consensus().agree_timings(local))
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, repr(winner.key()))
    assert len(set(gathered)) == 1, f"ranks chose different configs: {gathered}"
    assert math.isfinite(1.0) and torch is not None  # keep the imports honest
