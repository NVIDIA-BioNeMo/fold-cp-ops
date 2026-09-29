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

"""FAIL-QUICK per-config compile/run TIMEOUT GUARD (HARNESS stopgap).

WHY
---
The cut-keep cluster-family sweep DESYNC/FROZE because the COLD `cute.compile` of a cluster-family
config is slow AND grows with N, so the CROSS-RANK spread across the 16 ranks (lustre/ptxas contention)
exceeds the collective's tolerance:
  * `driver._barrier` == `rp._nvshmem_barrier` (NO timeout) -> an indefinite FREEZE (off-grid N=3008,
    48 min, process running).
  * `driver._consensus_ok` -> `da.all_reduce_max` == `torch.distributed.all_reduce(MAX)` (600 s
    dist-store timeout) -> a DistStoreError crash (on-grid N=4096, "6/16 clients joined").

This guard turns either silent multi-minute stall into a FAST, DIAGNOSED, structured skip -- the STOPGAP
while the REAL fix lands. The REAL fix is the CLAUDE.md HARD RULE: the cold compile MUST be <= 5 s (kill
the `range_constexpr`-with-nested-`if`-over-a-static-ceil(N/tile) blowup). A timeout is NOT a fix.

THE COLLECTIVE-SYMMETRY REQUIREMENT (load-bearing -- read `driver.py` _consensus_ok / run_cell)
------------------------------------------------------------------------------------------------
A PER-RANK LOCAL timeout that fires at DIFFERENT wall-clock times on different ranks would RE-CREATE the
exact desync (rank A aborts the config at T, rank B at T+5min -> the all_reduce still spreads >600 s).
So the budget is made COLLECTIVE by construction:
  1. `run_cell` already issues an UNCONDITIONAL all-rank barrier at every config boundary (driver.py:167)
     -> a COMMON T0 on every rank.
  2. Every rank arms the SAME `budget_s` from that common T0 -> the SIGALRM fires at ~T0+budget on every
     rank that overruns (barrier skew ~ms << budget). The ranks that overrun raise together; the ranks
     that finished are already at the consensus all_reduce -> the all_reduce completes within
     ~budget+epsilon on ALL ranks -> the outcome is consensus'd by the EXISTING `_consensus_ok`
     (build overrun -> local_ok=False -> global_ok=False -> ALL ranks skip the config symmetrically).

CAVEAT (honest): a Python SIGALRM can preempt an INTERRUPTIBLE stall (a CUDA/NCCL sync, a Python loop --
i.e. the RUN-hang case, and slow-but-yielding compiles), but it CANNOT preempt a pure-C++ MLIR/ptxas
compile blowup that never returns to the interpreter until it finishes. For THAT case the collective
budget cannot bound the slow rank in-process -> the only backstop is a RAISED dist-store timeout
(`raise_store_timeout` below, the coordinator's stopgap) so the job degrades to slow-not-crash while the
<= 5 s compile fix lands. Both are stopgaps; the compile fix is the resolution.
"""
import os
import signal
import threading


class PhaseTimeout(Exception):
    """Raised inside `phase_deadline` when the per-config budget is exceeded (a build/run overran)."""


def budget_s(env_name, default):
    """Read a timeout budget (seconds) from env; 0/<=0 disables the guard for that phase."""
    try:
        v = float(os.environ.get(env_name, default))
    except (TypeError, ValueError):
        v = float(default)
    return max(0.0, v)


class phase_deadline:
    """SIGALRM-armed per-phase budget. No-op if budget<=0 or not on the main thread (signal restriction).

    Usage (COLLECTIVE only when every rank enters from the SAME barrier'd T0 with the SAME budget):
        with phase_deadline(build_budget, "build", N=N, cfg=cfg):
            handle = target.build(ctx)     # -> PhaseTimeout on overrun -> caught by the driver's
                                           #    existing build-except -> consensus'd symmetric skip.
    """

    def __init__(self, seconds, label, **ctx):
        self.seconds = float(seconds or 0.0)
        self.label = label
        self.ctx = ctx
        self._prev = None
        self._armed = False

    def __enter__(self):
        if self.seconds > 0 and threading.current_thread() is threading.main_thread():
            self._prev = signal.signal(signal.SIGALRM, self._fire)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self._armed = True
        return self

    def __exit__(self, *exc):
        if self._armed:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if self._prev is not None:
                signal.signal(signal.SIGALRM, self._prev)
            self._armed = False
        return False  # never swallow -- let PhaseTimeout (or the real exception) propagate

    def _fire(self, signum, frame):
        det = " ".join(f"{k}={v}" for k, v in self.ctx.items())
        raise PhaseTimeout(
            f"{self.label} exceeded {self.seconds:.0f}s budget ({det}); STOPGAP fail-quick skip -- the "
            f"REAL fix is the <=5s cold-compile HARD RULE (kill the range_constexpr/ceil(N/tile) blowup)."
        )


def raise_store_timeout(seconds=None):
    """STOPGAP: raise the torch.distributed default-PG timeout so a slow (not-hung) cluster-family
    compile degrades to slow-not-CRASH instead of the 600 s DistStoreError desync, while the <=5 s
    compile fix lands. Best-effort (torch-version dependent); returns the applied seconds or None.

    The coordinator should ALSO pass the same timeout at process-group INIT (init_process_group/
    DistributedManager) -- this call only re-times an ALREADY-created default PG. Env override:
    CPO_HARNESS_STORE_TIMEOUT_S (default 3600). Set to 0 to skip.
    """
    import datetime
    if seconds is None:
        seconds = budget_s("CPO_HARNESS_STORE_TIMEOUT_S", 3600.0)
    if seconds <= 0:
        return None
    try:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return None
        td = datetime.timedelta(seconds=float(seconds))
        # torch 2.x: the private setter re-times the default PG's store; guard for API drift.
        from torch.distributed.distributed_c10d import _set_pg_timeout
        _set_pg_timeout(td, dist.group.WORLD)
        return float(seconds)
    except Exception:
        return None
