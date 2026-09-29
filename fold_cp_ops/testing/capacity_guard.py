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

"""Turn a device-capacity failure into a skip the WHOLE group takes.

CLAUDE.md requires the large end of a shape ladder to be bounded by a RUNTIME out-of-memory skip and
never by a memory-estimate gate: an estimate is a second implementation of the allocator, wrong in
both directions -- it hides a cell that would have fit and invents a limit nobody measured.

This module is that skip, in the one spelling that is safe under ``tests/distributed/``. It lives
beside :mod:`fold_cp_ops.testing.collective_guard` because HOW it calls ``gated_skip`` is its entire
correctness argument, and splitting the pair would put the collective-safety half outside the module
that enforces collective safety.

Three things it exists to fix, each measured on the 2026-08-30 distributed gate:

* **Two heaps, two exception types.** Operands land on the caching allocator and the recv lands on
  the SYMMETRIC heap, and they report exhaustion differently -- ``torch.cuda.OutOfMemoryError``
  versus ``RuntimeError: nvshmem_malloc failed``. A guard catching only the first left the top of
  the ladder FAILING rather than skipping.
* **A capacity predicate is not rank-invariant.** Two ranks with identical shapes differ by whatever
  else is resident on their device or symmetric heap, so one can raise while its peers do not. A
  unilateral skip there is a DEADLOCK, not a skip -- and worse, the ranks that did allocate go on to
  store into a peer buffer that no longer exists, which is an illegal memory access that poisons the
  context for every later cell. That is how two failures became eleven.
* **A failed test is memory-toxic.** pytest retains a failing test's frame for its traceback, so its
  tensors survive every teardown. Converting a capacity failure into a skip is containment, not
  cosmetics.
"""

from collections.abc import Callable
from typing import Any

import torch

from fold_cp_ops.testing.collective_guard import gated_skip

__all__ = ["HEAP_EXHAUSTED", "capacity_gate", "is_capacity_error"]

#: Message fragments that mean "the symmetric heap could not satisfy this".
#:
#: ``RuntimeError: nvshmem_malloc failed`` comes from a single ``TORCH_CHECK(ptr != nullptr)``, so it
#: says only that the pointer was null -- never why. It has TWO unrelated causes: a DEAD LIBRARY
#: after a finalize, and ordinary CAPACITY. The ``cuMemCreate failed`` / ``mem_heap.cpp`` lines on
#: stderr are what distinguish the second, e.g. at ``N_token=12288, D=256, cp=2`` where the recv is
#: 77 GB::
#:
#:     RuntimeError: nvshmem_malloc failed
#:     .../src/host/mem/mem_heap.cpp:2056: cuMemCreate failed
#:     .../src/host/mem/mem_heap.cpp:2135: error status: 7 (NVSHMEMX_ERROR_INTERNAL)
#:
#: Matched on the MESSAGE, never on ``RuntimeError`` itself: a bare ``except RuntimeError`` around a
#: build swallows every real failure the gate exists to surface, which is the same defect class as
#: widening a tolerance band until a cell passes.
HEAP_EXHAUSTED = ("nvshmem_malloc failed", "out of memory", "cuMemCreate failed")


def is_capacity_error(exc: BaseException) -> bool:
    """Whether an exception means "this cell does not fit", as opposed to "this cell is broken".

    Args:
        exc: The exception raised by an allocation or build step. Any type; only
            ``torch.cuda.OutOfMemoryError`` and a ``RuntimeError`` whose message carries one of
            :data:`HEAP_EXHAUSTED` count. Anything else is a defect, not a limit, and must propagate.

    Returns:
        True for a capacity failure on EITHER heap -- the caching allocator's or nvshmem's symmetric
        one. False for everything else.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and any(m in str(exc) for m in HEAP_EXHAUSTED)


def capacity_gate(what: str, fn: Callable[[], Any]) -> Any:
    """Run an allocating/compiling step, turning a capacity failure into a group-wide skip.

    Semantics
        ``fn`` is called inside ``try``. A capacity failure is CAPTURED rather than re-raised, the
        allocator's cache is dropped so the next cell in this process starts clean, and then EVERY
        rank -- the ones that raised and the ones that did not -- reaches the single ``gated_skip``
        call. **Reaching it unconditionally is the property that makes the reduction safe**; an early
        ``return`` on the success path would leave the raising ranks alone in an ``all_reduce`` their
        peers never enter, which is a hang or a mismatched collective rather than a skip. Any
        NON-capacity exception is re-raised untouched: a compile error is a failure, not a limit.

    Why the whole allocating region must be inside ``fn``
        Scope is as much a part of this guard as the exception types. On the gate that motivated this
        module the guard covered ``_build_inputs`` but ended before ``assert_written``, which
        allocates TWO reference buffers of the output size -- so the largest allocation in the cell
        sat outside it and failed rather than skipped. Wrap everything from the first allocation to
        the last, not just the obvious one.

    Args:
        what: Short label naming the step, e.g. ``"recv (symmetric)"``. It appears in the skip
            reason, so it should say WHICH allocation ran out, not merely that one did.
        fn: Zero-argument callable performing the allocation or build. Must be safe to abandon
            mid-way -- on a capacity failure its partial allocations are dropped on the floor and
            reclaimed, so anything needing an explicit release must do it in its own ``except``.

    Returns:
        Whatever ``fn`` returned, when no rank ran out of memory. Otherwise never returns.

    Raises:
        Skipped: via ``gated_skip``, on every rank, when ANY rank hit a capacity limit.
        Exception: anything ``fn`` raises that is not a capacity failure, unchanged.
    """
    result = None
    local_reason = None
    try:
        result = fn()
    except Exception as exc:  # re-raised below unless it is a capacity failure
        if not is_capacity_error(exc):
            raise
        local_reason = f"{what} did not fit: {type(exc).__name__}: {str(exc)[:200]}"
        torch.cuda.empty_cache()
    # Unconditional and COLLECTIVE: reached on the success path too, which is what keeps the
    # all_reduce inside it symmetric. Returns None when the group agreed nobody skips.
    gated_skip(local_reason)
    return result
