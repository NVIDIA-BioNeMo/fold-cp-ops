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

"""Choose a rendezvous port WITHOUT baking a port number into this repository.

Why this module exists
    A hardcoded port band is a LAUNCH-ENVIRONMENT policy smuggled into library code, the same
    class of mistake CLAUDE.md forbids for NIC/HCA/rail selection: it works on the machine it was
    written against and fails silently elsewhere. Two bands were baked in before this module --
    ``20000 + (jid % 20000)`` in `DistributedManager` (inherited verbatim from the upstream) and
    ``24000 + (cksum % 20000)`` in the bench cell runner -- and BOTH span the ephemeral range.

    Measured, on the same cell of the same sweep: the arm whose port landed at 42176 died with
    ``EADDRINUSE`` while the arm at 29710 ran in 38 s. 42176 is inside this node's
    ``ip_local_port_range`` (32768-60999), so any ordinary outbound connection could already own
    it. The failure surfaces as rank 0 never listening, i.e. the cell burns its whole timeout with
    only its banner logged -- which reads as a hang, not as a port collision.

The policy, in one line
    **Honour an operator-supplied port; otherwise derive one BELOW the running kernel's ephemeral
    floor, and PROVE it is free before returning it.** No port number is written in this file.

The floor is read from the OS, never assumed
    ``/proc/sys/net/ipv4/ip_local_port_range`` is the authority and it differs across hosts, so
    reading it at runtime is the difference between a portable rule and a second hardcoded band
    wearing a comment. When it cannot be read (non-Linux, restricted container) `ephemeral_floor`
    returns None and the caller falls back to a conservative default.

Why a probe and not just arithmetic
    A derived port is DETERMINISTIC per job/tag, so a retry lands on the port its own dead
    predecessor still holds in ``TIME_WAIT`` -- the recorded "a retry hits its own corpse" trap.
    Arithmetic cannot see that; a bind attempt can.
"""

from __future__ import annotations

import errno
import socket

#: Lowest port this module will ever hand out. Not a rendezvous port and not tunable policy -- it
#: is the privileged-port boundary, below which ``bind`` requires root regardless of what is free.
_UNPRIVILEGED_MIN = 1024


def ephemeral_floor(path: str = "/proc/sys/net/ipv4/ip_local_port_range") -> int | None:
    """Read the running kernel's lowest EPHEMERAL port, the boundary a rendezvous must stay below.

    Args:
        path: The sysctl file to read. Overridable for tests ONLY; production callers must use the
            default, because a caller-supplied path is a second place for a wrong value to live.

    Returns:
        The first field of ``ip_local_port_range`` (e.g. 32768), or None when the file is absent
        or unparseable -- non-Linux, or a container that masks ``/proc/sys``. None means "unknown",
        NOT "no floor": callers must degrade to probing rather than assume a default, since
        assuming one reintroduces exactly the baked constant this module exists to remove.
    """
    try:
        with open(path) as fh:
            lo = fh.read().split()[0]
        floor = int(lo)
    except (OSError, ValueError, IndexError):
        return None
    return floor if floor > _UNPRIVILEGED_MIN else None


def is_port_free(port: int, host: str = "") -> bool:
    """Probe whether ``port`` can actually be bound right now.

    **NO PRODUCTION CALLER, retained deliberately -- do not wire it into rendezvous selection.**
    Its only caller was ``pick_rendezvous_port``, removed in `eaf5e8d` because probing is
    self-defeating for a rendezvous: every rank probes the same candidate concurrently, one wins the
    bind and the rest see ``EADDRINUSE``, so the probe MANUFACTURES the collision it tests for, and
    those ranks then either walk to other candidates or raise -- both split the group. A rendezvous
    port must be a pure function of a job-uniform seed, which is what ``_initialize_slurm`` derives.
    Kept because the predicate is correct and tested and a future non-rendezvous caller (a
    single-process tool picking a debug port) is a reasonable use; deleting it is an owner call, not
    a cleanup.

    Semantics
        Binds a throwaway TCP socket and closes it. ``SO_REUSEADDR`` is deliberately NOT set: with
        it, a bind SUCCEEDS against a socket lingering in ``TIME_WAIT``, so the probe would report
        free exactly in the case this function exists to catch -- a retry of a cell whose own
        predecessor just died. The check is inherently racy (a port free now may be taken a
        millisecond later); it narrows the window, it does not close it.

    Args:
        port: Port to probe. A value outside 1..65535 returns False rather than raising, so a
            caller walking a range does not have to bounds-check every candidate.
        host: Interface to bind. Default "" means INADDR_ANY, which is what a rendezvous listener
            uses; probing a single interface would pass while the real bind fails.

    Returns:
        True if the bind succeeded. False on ``EADDRINUSE``/``EACCES``/``EADDRNOTAVAIL`` or an
        out-of-range port.
    """
    if not (_UNPRIVILEGED_MIN <= port <= 65535):
        return False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, errno.EACCES, errno.EADDRNOTAVAIL):
            return False
        raise
    finally:
        s.close()
