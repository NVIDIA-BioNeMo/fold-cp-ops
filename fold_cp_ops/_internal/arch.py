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

# Copyright (c) 2025, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""The SM90 capability boundary, and the one sentence it is announced with.

`fold-cp-ops` ships SM90 (H100/H200) kernels **only** — every MMA atom under `kernels/` is a Hopper
WGMMA atom that cutlass-dsl hard-gates to `sm_90a`. That is a decision, not an oversight, so a
non-SM90 device has to be refused **at the front door** with something the caller can act on. Left
as an arch branch, a B200 call would instead die several frames into dispatch on a `NameError` for
an SM100 class this package does not contain.

Everything here is host-side Python: it queries a device capability, compares it to one constant,
and either raises or returns. Nothing is traced, nothing is emitted. It lives in `_internal/`
rather than `compile_time/` for that reason — `compile_time/` is for metaprograms that emit layout
and type algebra during tracing, and this module emits nothing at all.

`get_device_capacity` honours the `CPO_ARCH` environment override, which is what lets the
CPU-only tests exercise the boundary (compile for `sm_100a` on a box that has no Blackwell in it,
and assert the refusal). The override is read on every *uncached* call, so a test that flips it
must call `get_device_capacity.cache_clear()`.
"""

import os
import re
from functools import lru_cache
from typing import Optional, Tuple

import torch

import cutlass

#: The one architecture `fold-cp-ops` ships kernels for. Not a tunable: raising it would not make
#: an SM100 kernel exist, it would only move the failure from this module into MLIR generation.
SUPPORTED_ARCH_MAJOR = 9


class UnsupportedArchError(RuntimeError):
    """Raised when an entry point is called on a GPU that is not SM90.

    A distinct type — not a bare `RuntimeError`, and deliberately not `AssertionError` — so a caller
    can route around the capability boundary (fall back to a torch reference, skip a test) by
    catching it, rather than by matching on the message text. `AssertionError` would additionally be
    wrong here because `python -O` strips `assert`, which would delete the boundary entirely.
    """


@lru_cache
def get_max_active_clusters(cluster_size: int) -> int:
    """Occupancy of the current device in clusters, for the persistent-scheduler grid.

    A persistent GEMM launches exactly as many clusters as the device can hold resident and then
    loops over work tiles, so this number *is* the grid size. Cached because the query goes through
    the CUDA occupancy API and the answer cannot change within a process.

    Args:
        cluster_size: Threadblocks per cluster, i.e. `cluster_M * cluster_N`. Must be a positive
            power of two no greater than 8 — the hardware cluster limit. A larger or non-power-of-two
            value is rejected by the underlying CUTLASS query, not here.

    Returns:
        The maximum number of simultaneously-resident clusters of that size.

    Raises:
        RuntimeError: If no CUDA device is available. This is a device query, so unlike the rest of
            this module it cannot be exercised via `CPO_ARCH`.
    """
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


def _parse_arch_str(arch_str: str) -> Tuple[int, int]:
    """Parse an architecture string into a `(major, minor)` capability pair.

    Args:
        arch_str: One of the spellings CUDA tooling uses interchangeably — `"90"`, `"sm90"`,
            `"sm_90"`, `"sm_90a"`, `"sm_100a"`. Surrounding whitespace is stripped and the match is
            case-insensitive. The trailing feature letter (`a`/`f`) is accepted and discarded: it
            selects an ISA variant, not a capability.

    Returns:
        `(major, minor)`, e.g. `("sm_90a")` -> `(9, 0)` and `("sm_100a")` -> `(10, 0)`. Note the
        split is "all but the last digit" / "last digit", which is how `100` becomes `(10, 0)`
        rather than `(1, 0)`.

    Raises:
        ValueError: If the string does not match, naming the variable so the reader knows it came
            from `CPO_ARCH` and not from a device query.
    """
    match = re.match(r"^(?:sm_?)?(\d+)(\d)([af]?)$", arch_str.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid CPO_ARCH format: {arch_str!r} (expected e.g. '90', 'sm_90')")
    major, minor, _ = match.groups()
    return int(major), int(minor)


@lru_cache
def get_device_capacity(device: Optional[torch.device] = None) -> Tuple[int, int]:
    """The `(major, minor)` compute capability a kernel should be compiled and gated for.

    Args:
        device: The device to query, or None for the current one. Used only as the cache key and as
            the argument to `torch.cuda.get_device_capability`; ignored entirely when `CPO_ARCH` is
            set.

    Returns:
        `(major, minor)`, e.g. `(9, 0)` on an H100.

    Raises:
        ValueError: If `CPO_ARCH` is set to something `_parse_arch_str` cannot read.
        RuntimeError: If `CPO_ARCH` is unset and no CUDA device is available.

    Note:
        `CPO_ARCH` overrides the query entirely, which is what makes the capability boundary
        testable without the silicon — set it to `"sm_100a"` on an H100 and the front-door check
        must refuse. Because the result is `lru_cache`d, a test that changes the variable must call
        `get_device_capacity.cache_clear()` on both sides of the change or it will read a stale
        answer.
    """
    arch_override = os.environ.get("CPO_ARCH")
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    return torch.cuda.get_device_capability(device)


def check_arch_supported(device_capacity: Tuple[int, int]) -> None:
    """Refuse a non-SM90 capability, with the boundary's single canonical message.

    Takes an already-queried capability rather than querying itself, so an entry that was keyed and
    compiled on one can re-check it without a second device round-trip — and so a `@jit_cache`d
    compile step can re-validate its own cache key instead of trusting it.

    Args:
        device_capacity: A `(major, minor)` pair, normally from `get_device_capacity`. Only the
            major is compared; the minor is used solely to render the message.

    Returns:
        None. Returning normally *is* the "supported" answer.

    Raises:
        UnsupportedArchError: If `device_capacity[0] != SUPPORTED_ARCH_MAJOR`. The message is
            written here and nowhere else: one capability boundary, one sentence, so it cannot drift
            between the entry points that guard it.
    """
    if device_capacity[0] != SUPPORTED_ARCH_MAJOR:
        raise UnsupportedArchError(
            f"fold-cp-ops requires SM90 (H100/H200); got "
            f"sm_{device_capacity[0]}{device_capacity[1]}. The SM100/SM120 GEMM kernels are not "
            f"part of this package."
        )


def require_sm90(device: Optional[torch.device] = None) -> Tuple[int, int]:
    """Query the device and refuse it if it is not SM90. The front door itself.

    Call this **before** selecting a kernel class or branching on a capability — that ordering is
    the whole point. Once dispatch has begun, an unsupported device surfaces as an attribute lookup
    on a name that was deleted, which tells the caller nothing.

    Args:
        device: The device to gate on, or None for the current one. Pass the device the *tensors*
            live on, not the default one, or a multi-GPU process can validate the wrong GPU.

    Returns:
        The `(major, minor)` capability, so the caller can key a compile cache on it without a
        second query.

    Raises:
        UnsupportedArchError: If the device is not SM90.
        ValueError: If `CPO_ARCH` is set to an unparseable string.
        RuntimeError: If `CPO_ARCH` is unset and there is no CUDA device.
    """
    cap = get_device_capacity(device)
    check_arch_supported(cap)
    return cap
