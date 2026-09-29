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

"""GPU clock-lock util for reproducible perf-gate benchmarks.

Locks the graphics AND memory clocks to the device max BEFORE timing and resets
them AFTER, so the ±10% perf gate is not tripped by DVFS / boost drift between
runs. Used by the ``tests/benchmark_test/`` perf-gate tests (via the autouse
``clock_locked`` session fixture in ``conftest.py``).

MANDATORY FALLBACK (task requirement): clock control needs root / an enabled
persistence daemon on most clusters, so ``nvidia-smi --lock-gpu-clocks`` often
returns *permission denied* for a non-root user. When ANY step fails (binary
missing, query fails, permission denied) we ``logging.warning`` once and PROCEED
UNLOCKED — the ±10% gate still applies. This util NEVER hard-fails on
lock-unavailable; a failed lock is a warning, not an error.

Interface (all functions are best-effort + exception-safe):
  * ``query_max_clocks(index)``  -> ``(gr_mhz, mem_mhz)`` or ``None``.
  * ``lock_clocks(indices=None)`` -> the list of indices actually locked (``[]``
    if none — the fallback path).
  * ``reset_clocks(indices)``    -> reset gr+mem clocks on those indices.
  * ``clock_lock(indices=None)``  -> context manager: lock on enter, reset on exit.

One-locker-per-node: under torchrun / srun EVERY rank imports this, so only
``LOCAL_RANK`` 0 (or a non-distributed launch) actually touches the clocks; the
lock is global per physical GPU, so a single locker covers every rank on the node.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

_NVSMI = shutil.which("nvidia-smi") or "nvidia-smi"
_WARN_TAG = "-> perf may fluctuate; ±10% gate still applies"


def _run(args, timeout: int = 30):
    """Run ``nvidia-smi <args>``; return the CompletedProcess (never raises for a nonzero rc)."""
    return subprocess.run(
        [_NVSMI, *args], capture_output=True, text=True, timeout=timeout, check=False
    )


# Failure signatures for a clock command (some drivers PRINT the error but still exit 0, so we scan the
# output too — not only the return code). A SUCCESS prints e.g. 'GPU clocks set to ...'.
_FAIL_SIGS = (
    "not have permission",
    "permission",
    "terminating early",
    "invalid combination",
    "failed",
)


def _looks_failed(r) -> bool:
    """True if the nvidia-smi clock command failed (nonzero rc OR an error signature in its output)."""
    if r.returncode != 0:
        return True
    blob = ((r.stdout or "") + (r.stderr or "")).lower()
    return any(sig in blob for sig in _FAIL_SIGS)


def _first_line(r) -> str:
    detail = ((r.stderr or "") or (r.stdout or "")).strip().splitlines()
    return detail[0] if detail else "nonzero rc"


def query_max_clocks(index: int = 0):
    """Return ``(max_gr_mhz, max_mem_mhz)`` for physical GPU ``index``, or ``None`` on any failure."""
    try:
        r = _run(
            [
                "--query-gpu=clocks.max.graphics,clocks.max.memory",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ]
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        gr_s, mem_s = r.stdout.strip().splitlines()[0].split(",")
        return int(gr_s.strip()), int(mem_s.strip())
    except Exception as e:  # noqa: BLE001 — best-effort: any failure -> unlocked fallback
        logger.warning("clock-lock: max-clock query failed on GPU %s: %r", index, e)
        return None


def _enumerate_indices():
    """Physical GPU indices visible to the driver (``[0]`` on any failure)."""
    try:
        r = _run(["--query-gpu=index", "--format=csv,noheader"])
        if r.returncode == 0:
            idxs = [int(x.strip()) for x in r.stdout.split() if x.strip().isdigit()]
            if idxs:
                return idxs
    except Exception:  # noqa: BLE001
        pass
    return [0]


def _target_indices(indices=None):
    """Which physical GPUs to manage: caller override, else CUDA_VISIBLE_DEVICES (index form), else all.

    Honoring CUDA_VISIBLE_DEVICES (when it is a plain index list) keeps us a good
    cluster citizen on a partial-node alloc; a UUID-form CVD or an unset one falls
    back to every visible GPU (correct on the whole-node H100 allocs these pins
    are measured on).
    """
    if indices is not None:
        return list(indices)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        if parts and all(p.isdigit() for p in parts):
            return [int(p) for p in parts]
    return _enumerate_indices()


def _only_local_rank0() -> bool:
    """True unless this is a non-zero ``LOCAL_RANK`` (one locker per node under torchrun/srun)."""
    lr = os.environ.get("LOCAL_RANK")
    return lr is None or lr == "0"


def lock_clocks(indices=None):
    """Lock gr+mem clocks to max on the target GPUs. Return the list actually locked (``[]`` = fallback).

    Best-effort: a per-GPU query or lock failure (permission denied, no persistence
    daemon) is warned once and skipped — that GPU stays unlocked and the ±10% gate
    still applies. A non-``LOCAL_RANK``-0 rank is a no-op (a peer locks the node).

    ``CPO_CLOCKLOCK_DISABLE=1`` forces the unlocked path (returns ``[]`` without any
    nvidia-smi call) — used for a FAIR cross-cluster/hardware comparison (e.g. venue B, where
    a non-root user CAN lock the memory clock, vs venue A, where it cannot) and for the
    locked-vs-unlocked A/B experiment.
    """
    if os.environ.get("CPO_CLOCKLOCK_DISABLE") == "1":
        logger.info("clock-lock: DISABLED via CPO_CLOCKLOCK_DISABLE -> running unlocked")
        return []
    if not _only_local_rank0():
        return []
    locked = []
    for i in _target_indices(indices):
        mc = query_max_clocks(i)
        if mc is None:
            logger.warning("GPU clock-lock unavailable on GPU %d (query failed) %s", i, _WARN_TAG)
            continue
        gr, mem = mc
        # nvidia-smi allows only ONE device modification per invocation ("Invalid combination of input
        # arguments. Only one device modification may be done at a time."), so gr and mem clocks MUST be
        # locked in SEPARATE calls. A gr-only lock (mem denied) still stabilizes the compute clock, so we
        # keep the index if EITHER succeeds and reset both kinds on it later.
        any_ok = False
        try:
            rg = _run(["-i", str(i), f"--lock-gpu-clocks={gr}"])
            if _looks_failed(rg):
                logger.warning(
                    "GPU graphics-clock lock unavailable on GPU %d (%s) %s",
                    i,
                    _first_line(rg),
                    _WARN_TAG,
                )
            else:
                any_ok = True
            rm = _run(["-i", str(i), f"--lock-memory-clocks={mem}"])
            if _looks_failed(rm):
                logger.warning(
                    "GPU memory-clock lock unavailable on GPU %d (%s) %s",
                    i,
                    _first_line(rm),
                    _WARN_TAG,
                )
            else:
                any_ok = True
        except Exception as e:  # noqa: BLE001
            logger.warning("GPU clock-lock unavailable on GPU %d (%r) %s", i, e, _WARN_TAG)
        if any_ok:
            locked.append(i)
    if locked:
        logger.info("clock-lock: locked clocks to max on GPUs %s", locked)
    return locked


def reset_clocks(indices) -> None:
    """Reset gr+mem clocks to default on ``indices`` (best-effort; separate calls, only what lock returned)."""
    for i in indices:
        for arg in ("--reset-gpu-clocks", "--reset-memory-clocks"):
            try:
                _run(["-i", str(i), arg])
            except Exception as e:  # noqa: BLE001
                logger.warning("clock-lock: reset (%s) failed on GPU %d: %r", arg, i, e)


@contextlib.contextmanager
def clock_lock(indices=None):
    """Context manager: lock clocks on enter (best-effort), reset on exit. Yields the locked-index list."""
    locked = lock_clocks(indices)
    try:
        yield locked
    finally:
        reset_clocks(locked)
