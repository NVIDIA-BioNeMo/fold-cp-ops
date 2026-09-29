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

"""Making every rank choose the SAME config -- the thing the upstream harness had no notion of.

Autotuning a kernel that contains a collective is not autotuning-with-extra-steps; it is a different
problem, and the upstream harness solves none of it. There, each rank independently times its own
configs and takes ``min(timings)``. Three ways that goes wrong, none of which raises:

1. **Divergent winners.** Two ranks measure the same two configs at 1.00 and 1.01 ms, in opposite
   orders because of noise. They pick different kernels. The next collective has ranks in different
   kernels with different tile counts -- a hang, surfacing minutes later as a timeout with no
   indication that autotuning caused it.
2. **Divergent candidate sets.** A config that raises on one rank is scored ``inf`` THERE and stays
   in contention elsewhere. The rank that failed has already skipped the collective the others are
   still waiting in.
3. **Meaningless timings.** A rank that finishes its part of a collective early records a fast time
   that is really its peers' slowness. The per-rank minimum optimizes for whoever was luckiest.

The fixes here are all "make the ranks agree, by construction rather than by luck":

* :meth:`Consensus.agree_failures` unions the failures, so a config that failed ANYWHERE is dropped
  EVERYWHERE, before it can be scored.
* :meth:`Consensus.agree_timings` takes the **max** over ranks -- the slowest PE is the collective's
  real cost, and it is the same number on every rank.
* :meth:`Consensus.pick` breaks ties on the config's canonical key, not on dict order, so identical
  inputs give identical winners even when two configs measure exactly equal.

**Single-process runs pay nothing.** With no initialized process group every method is the identity,
so a local kernel takes the same path it always did and needs no separate code.
"""

import math
import os
from typing import Dict, Optional, Tuple

from fold_cp_ops._internal.autotune.config import AutotuneConfig


class Consensus:
    """Reconciles per-rank autotuning results into one answer every rank agrees on.

    Args:
        group: An optional ``torch.distributed`` process group. ``None`` (the default) means "use
            the default group if one is initialized, otherwise run single-process". Pass an
            explicit group when the kernel's collective runs on a sub-group -- the consensus must
            span exactly the ranks that participate in the kernel's collective, and no others.
            Reconciling across a WIDER group than the kernel uses is not merely wasteful: it makes
            the tuner's own collectives a second synchronization the kernel does not have.

    Attributes:
        enabled: Whether reconciliation actually happens. False in a single process, which is the
            case every local kernel takes.
    """

    def __init__(self, group: Optional[object] = None):
        self._group = group
        self._dist = None
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                self._dist = dist
        except ImportError:  # pragma: no cover - torch always present in this package
            self._dist = None
        self.enabled = self._dist is not None

    @property
    def world_size(self) -> int:
        """Number of ranks being reconciled; 1 when disabled."""
        return self._dist.get_world_size(self._group) if self.enabled else 1

    @property
    def rank(self) -> int:
        """This process's rank; 0 when disabled."""
        return self._dist.get_rank(self._group) if self.enabled else 0

    def agree_timings(self, timings: Dict[AutotuneConfig, float]) -> Dict[AutotuneConfig, float]:
        """Reduce each config's timing to the SLOWEST rank's, and drop anything that failed anywhere.

        Purpose
            Two jobs in one collective, which is the point. A collective finishes when its last
            participant does, so the max IS what the config cost the job -- a rank that returned
            early was waiting, not fast. And because a failed candidate arrives here as ``inf``, the
            same max propagates that ``inf`` to every rank, so a config that failed anywhere is
            dropped everywhere. No separate failure exchange is needed.

        Semantics
            Collective, and the reduction MUST be reached by every rank whether or not its own
            measurement raised -- that is why the caller records ``inf`` rather than skipping the
            candidate. A rank that skipped this call leaves its peers blocked in the reduction.

            Under SPMD every rank runs the same config at the same time, so divergent failure means
            a setup bug (uneven shards, or a per-rank OOM) rather than a normal condition. It is
            therefore REPORTED: :meth:`divergent_failures` names the configs that failed on some
            ranks but not all, so the bug surfaces instead of being silently absorbed.

        Args:
            timings: This rank's measured milliseconds per config, with ``inf`` for any candidate
                that raised. Every rank must pass the same key set.

        Returns:
            The element-wise max across ranks, identical on every rank, with non-finite entries
            (i.e. failed-anywhere configs) removed. Returns the finite subset unchanged when
            single-process.

        Raises:
            RuntimeError: If the ranks disagree about which configs were measured -- reducing over
                mismatched sets would compare timings for different kernels.
        """
        if not self.enabled:
            self._divergent = ()
            return {c: t for c, t in timings.items() if math.isfinite(t)}
        keys = sorted(timings, key=lambda c: c.key())
        gathered = [None] * self.world_size
        self._dist.all_gather_object(gathered, [c.key() for c in keys], group=self._group)
        expected = gathered[0]
        for r, part in enumerate(gathered):
            if part != expected:
                raise RuntimeError(
                    f"autotune ranks disagree about which configs were measured (rank {r} differs "
                    f"from rank 0). Every rank must derive its candidate list from the same pure "
                    f"function of the request; a validity rule that reads the device, an env var "
                    f"or a global will do this. Reducing over mismatched sets would compare "
                    f"timings for different kernels."
                )
        import torch

        local = torch.tensor(
            [timings[c] for c in keys],
            dtype=torch.float64,
            device="cuda" if self._backend_is_nccl() else "cpu",
        )
        # MIN alongside MAX is what makes divergence VISIBLE: a config that failed on some ranks and
        # not others has min finite and max infinite. One extra reduction, and it turns a silent
        # setup bug into a named one.
        worst = local.clone()
        best = local.clone()
        self._dist.all_reduce(worst, op=self._dist.ReduceOp.MAX, group=self._group)
        self._dist.all_reduce(best, op=self._dist.ReduceOp.MIN, group=self._group)
        self._divergent = tuple(
            c
            for c, hi, lo in zip(keys, worst.tolist(), best.tolist())
            if not math.isfinite(hi) and math.isfinite(lo)
        )
        return {c: float(v) for c, v in zip(keys, worst.tolist()) if math.isfinite(v)}

    def divergent_failures(self) -> Tuple[AutotuneConfig, ...]:
        """Configs that failed on SOME ranks but not all, from the most recent reduction.

        Purpose
            Under SPMD this should always be empty. A non-empty result is a setup bug worth naming
            -- uneven local shards, or one rank running out of memory -- and it is exactly the
            condition that used to become a hang rather than a message.

        Returns:
            The diverging configs, in canonical order. Empty when single-process, when nothing
            diverged, or before any reduction has run.
        """
        return getattr(self, "_divergent", ())

    def _backend_is_nccl(self) -> bool:
        """Whether the reduction tensor must live on the GPU.

        NCCL cannot reduce a CPU tensor and gloo cannot reduce a CUDA one, so the placement is not
        a preference -- getting it wrong is an immediate backend error.
        """
        try:
            return "nccl" in str(self._dist.get_backend(self._group)).lower()
        except Exception:
            return False

    @staticmethod
    def pick(timings: Dict[AutotuneConfig, float]) -> AutotuneConfig:
        """Choose the winner deterministically: fastest, ties broken by canonical key.

        Purpose
            ``min(timings, key=timings.get)`` returns whichever equal-valued key the dict yields
            first, which depends on insertion order. After :meth:`agree_timings` every rank holds
            identical values, so an order-dependent tie-break is the LAST place ranks can still
            diverge -- and exact ties are common once timings are quantized by a reduction.

        Args:
            timings: Config to milliseconds. Must be non-empty.

        Returns:
            The winning config. Identical on every rank given identical input.

        Raises:
            ValueError: If `timings` is empty, or every entry is non-finite (i.e. every config
                failed) -- which must be a loud failure rather than an arbitrary pick.
        """
        if not timings:
            raise ValueError("no timings to choose from")
        finite = {c: t for c, t in timings.items() if t == t and t != float("inf")}
        if not finite:
            raise ValueError(
                f"every candidate config failed to run ({len(timings)} tried). Autotuning cannot "
                f"select a winner; re-run with CPO_AUTOTUNE_VERBOSE=1 to see each config's error."
            )
        return min(sorted(finite), key=lambda c: (finite[c], c.key()))

    def barrier(self) -> None:
        """Synchronize the ranks; a no-op single-process.

        Used to bracket the measurement phase so that one rank's compilation does not land inside
        another rank's timed region -- a skew that silently inflates whichever config happened to
        be measured first.
        """
        if self.enabled:
            self._dist.barrier(group=self._group)

    @staticmethod
    def is_distributed_env() -> bool:
        """Whether the process was launched under a MULTI-RANK distributed launcher, group or not.

        Distinct from :attr:`enabled`, which asks whether a group is INITIALIZED. The gap between
        the two is a real failure mode: a torchrun job that autotunes before ``init_process_group``
        would tune per-rank with no consensus and no warning, so the tuner checks this to refuse.

        **`WORLD_SIZE > 1` is required, and that clause is load-bearing.** The refusal it drives
        exists because "each rank would measure and choose independently, with no guarantee they
        agree, and a kernel containing a collective then deadlocks". **At one rank there is nothing
        to disagree with and no peer to deadlock against, so refusing there is vacuous** -- and it is
        not hypothetical: `DistributedManager._initialize_slurm` does
        `os.environ.setdefault("RANK", procid)` / `("WORLD_SIZE", ntasks)`, so ANY `srun --ntasks=1`
        sets `RANK=0`/`WORLD_SIZE=1` process-globally and permanently. Without this clause the
        single-device suite launched the way `CLAUDE.md` documents for a cluster
        (`srun ... pytest tests/ -n 8`) fails **84 tests**, every one of them a refusal to autotune a
        single-rank run. Measured 2026-08-21.

        Returns:
            True when the launcher describes MORE THAN ONE rank. False when the vars are absent or
            describe a single rank. An unparseable ``WORLD_SIZE`` returns True -- failing toward the
            refusal, because a loud error beats silently tuning per-rank on a real multi-rank job.
        """
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            return False
        try:
            return int(os.environ["WORLD_SIZE"]) > 1
        except ValueError:
            return True
