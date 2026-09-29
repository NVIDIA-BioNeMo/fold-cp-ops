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


"""The shape gate every distributed BenchTarget shares.

Lifted from the upstream's ``benchmark/distributed/back_a2a_store_bench.py::_shape_check``,
which is not ported here: the targets need this one 13-line predicate out of a 1225-line module.
It lives in the harness rather than in a target because `front_a2a`, `front_route2`,
`cluster_drain` and `trimul_e2e` all gate on it, and a target importing another target for a
predicate is how a plug-in seam stops being one. It does NOT live in ``bench_utils`` -- that
module's single subject is TIMING (it re-exports ``fold_cp_ops/_internal/bench_timing.py`` and
CLAUDE.md pins it as the timing life-line), and a shape predicate there would blur the one module
whose job is measurement.

Renamed from ``_shape_check`` to `shape_check` on the move: it is now imported across module
boundaries, and a leading underscore there misdescribes its visibility.
"""


def shape_check(N, cp0, cp1):
    """Whether a token extent is legal on a ``(cp0, cp1)`` mesh, and if not, why.

    Purpose
        The ONE shape gate shared by every distributed target, so a cell that cannot run reports a
        reason a human can act on instead of failing somewhere inside a descriptor.

    Semantics
        Arbitrary ``N`` and arbitrary ``cp`` -- including a 1-D mesh (``cp1 == 1``) and
        non-power-of-two cp -- are supported. The ONLY invalid case is the 16-byte alignment
        HARD RULE: ``N`` must shard on both axes AND each local extent must be a multiple of 8
        (bf16 -> 8 elements = 16 B, the TMA minimum). Pure integer arithmetic, no device access
        and no global state, which is what lets the driver call it on every rank and rely on the
        answer being identical everywhere -- that determinism is load-bearing for collective
        symmetry.

    Args:
        N: Token extent. Any positive integer; divisibility is checked, not assumed.
        cp0: Context-parallel extent of the i axis. Must be > 0.
        cp1: Context-parallel extent of the j axis, or 1 for a 1-D token shard. Must be > 0.

    Returns:
        ``(True, "")`` if the shape is legal, else ``(False, reason)`` with `reason` naming the
        extent that failed and what to pick instead.
    """
    if cp0 <= 0 or cp1 <= 0:
        return False, f"invalid cp mesh cp0={cp0} cp1={cp1}"
    if N % cp0 != 0 or N % cp1 != 0:
        return False, f"N={N} not divisible by cp0={cp0} and/or cp1={cp1} (not token-shardable)"
    ni, nj = N // cp0, N // cp1
    if ni % 8 != 0 or nj % 8 != 0:
        return False, (
            f"N_i_loc=N/cp0={ni} (%8={ni % 8}) / N_j_loc=N/cp1={nj} (%8={nj % 8}) not %8 == not "
            f"16-B-aligned (bf16 TMA HARD RULE); pick N with N/cp0 and N/cp1 both multiples of 8"
        )
    return True, ""
