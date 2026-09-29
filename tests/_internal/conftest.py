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

"""pytest fixtures for the ``fold_cp_ops/_internal/bench_timing.py`` tests.

The local (single-GPU) groups A-F in ``test_bench_timing.py`` need no fixture. The
distributed group G is *torchrun-collected*::

    CUDA_VISIBLE_DEVICES=0,1 CPO_CACHE_ENABLED=0 NVSHMEM_DISABLE_NVLS=1 \
    torchrun --nproc_per_node=2 -m pytest -q \
        tests/_internal/test_bench_timing.py -k distributed

Every torchrun rank is its own pytest process; all ranks form ONE
``torch.distributed`` process group initialized once per session by the
``dist_env`` fixture (mirrors the pattern in ``tests/distributed/conftest.py``).
When NOT launched under torchrun (``WORLD_SIZE`` unset or ``==1``) the fixture
``pytest.skip``s, so the same file runs clean locally (group G skipped).

Deliberately *unlike* ``tests/distributed/conftest.py`` this fixture does NOT
depend on ``DistributedManager``: ``bench_timing`` auto-resolves plain
``torch.distributed`` when called with ``dist=None`` (its single trusted path),
so the fixture initializes exactly that — the resolution actually under test.

**Scope note.** This file sits at ``tests/_internal/`` rather than beside its one consumer,
because the timing primitive moved into the package and the test moved with it (a source file is
tested at the mirrored path). So ``dist_env`` and the teardown hook below are now in scope for
EVERY ``tests/_internal`` module. Both are inert outside a torchrun launch: the fixture is only
constructed when a test requests it and skips when ``WORLD_SIZE <= 1``, and the hook returns
immediately unless a process group is live. Nothing else under ``tests/_internal`` requests either.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


def _under_torchrun() -> bool:
    """True only for a genuine multi-rank torchrun launch."""
    return (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


@pytest.fixture(scope="session")
def dist_env():
    """Init ``torch.distributed`` once/session from the torchrun env; skip if single-process.

    Pins ``cuda:LOCAL_RANK`` BEFORE ``init_process_group`` so NCCL binds each rank
    to a distinct GPU. Yields a namespace with ``rank`` / ``world_size`` /
    ``local_rank`` / ``device``; tears the group down behind a device-scoped
    barrier at session end.
    """
    if not _under_torchrun():
        pytest.skip("distributed bench tests require torchrun with WORLD_SIZE>1")

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("distributed bench tests require CUDA")

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.device_count() < 1:
        pytest.skip("no CUDA devices visible to this rank")

    # Pin the device before init so each rank's NCCL comm binds to a distinct GPU
    # (LOCAL_RANK indexes into CUDA_VISIBLE_DEVICES).
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    env = SimpleNamespace(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
    )
    assert env.world_size == world_size
    yield env

    # Teardown: barrier so no rank destroys its group while a peer still uses it.
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Barrier after each distributed test so ranks stay lockstep across group G.

    Without this, a rank that finishes test N quickly could enter test N+1 and
    issue a collective while a slow rank is still draining test N (a cross-test
    desync). No-op unless the process group is live, so local / skipped sessions
    and every non-distributed module under ``tests/_internal`` are unaffected. A teardown barrier
    must never mask the test's own outcome, which is why every failure here is swallowed.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
            else:
                dist.barrier()
    except Exception:
        pass
