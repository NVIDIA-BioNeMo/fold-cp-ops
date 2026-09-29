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
"""The shared TOPOLOGY guard for the distributed suites -- one probe, in one place.

Purpose
    A COUPLED peer store must not run on a job whose peers are reached over IB. This module decides
    that, once, for every suite that drives one. Two suites use it today
    (``test_gemm_a2a_epi.py``, ``test_dual_gated_gemm_staged_a2a.py``), which is exactly why it is a
    module: the alternative is a private copy per suite, and a suite that never grows one runs
    unguarded across nodes and takes a silent illegal memory access rather than a skip.

Why it wraps the store's own probe rather than re-implementing it
    The predicate is `fold_cp_ops.distributed.gemm_sm90_a2a.build_p2p_table` -- the NVSHMEM
    ``TEAM_SHARED`` probe the STORE itself uses to choose NVLink over IB. Guarding on the kernel's
    own source is the point: a divergent copy would guard on a different topology than the kernel
    routes on, and the two would disagree exactly on the mesh where it matters.

    That also fixes the layering: this module holds no topology knowledge of its own, so it cannot
    drift. It is a skip policy over somebody else's measurement.

Relationship to the kernel's front-door guard
    `GemmSm90A2A._guard_coupled_nvlink_only` REFUSES a coupled configuration on a cross-node mesh,
    raising at the front door. This module SKIPS the test cell that would have built one. The two
    are complementary and neither replaces the other: without the raise a user gets an unattributed
    fault, and without the skip a legitimate cross-node run turns red on cells whose path is
    NVLink-only by hardware. `tests/distributed/test_gemm_sm90_a2a.py` asserts the raise; this
    module keeps the suites green where the raise is correct.

Note
    The upstream's copy carried a provenance narrative arguing from a private guard in
    ``test_front_a2a_staged.py`` and naming a canonical file for it. Neither file exists in this
    tree, so the narrative is re-sourced above rather than carried; the ARGUMENT it made -- one
    copy, wrapping the store's own probe -- is unchanged, and is the reason this module exists.

    The upstream also carried a ``job_has_ib_peers_pe_map`` taking a ``PeMap``. It is deliberately
    NOT here: nothing in the upstream's own test tree calls it, and its body delegates to
    ``fused_trimul_autotune._probe_has_ib_peers``, a module this tree has not ported. Carrying it
    would import an absent module for a function with no callers. Restore it with its caller, not
    before.
"""

from fold_cp_ops.testing.collective_guard import rank_invariant_skip

# Why the predicate is rank-INVARIANT, in the upstream's own words -- carried verbatim because it is
# the argument, not a paraphrase of one. Every `rank_invariant_skip` below declares this.
_UNIFORM_BECAUSE = (
    "build_p2p_table is a pure function of pe_table, and the cp pe_table is a rank-INVARIANT job "
    "property, so every rank evaluates this predicate identically and skips (or does not skip) "
    "together -- no teardown-barrier desync. Only a predicate that can DIVERGE per rank (an OOM "
    "catch, a per-rank build failure) needs the all_reduce consensus of CollectiveGate"
)

# The pure-COUPLED peer store (a TMA-S2G / raw STG straight into the peer's symmetric heap via
# ``nvshmem_ptr``) is NVLink-ONLY **by hardware**: ``nvshmem_ptr`` returns NULL for an IB (non-P2P)
# peer, so a coupled store to a cross-node peer faults (CUDA_ERROR_ILLEGAL_ADDRESS, surfacing at the
# next barrier). That is a genuine CAPABILITY BOUNDARY of the coupled path, NOT a supported shape
# being skipped: the cross-node store is covered by the ib_drain / V6 differential paths, which stage
# IB peers through a symmetric ring drained by a blocking put. So a coupled peer-store test
# self-skips on a job WITH IB peers and still runs on every all-P2P (single-NVLink-domain, cp<=8)
# job -> no coverage gap.
COUPLED_NVLINK_ONLY_SKIP = (
    "coupled peer store is NVLink-only -- a cross-node (has_ib) job faults on NULL IB descriptors; "
    "cross-node coverage lives on the ib_drain / V6 differential path."
)


def job_has_ib_peers(pe_table) -> bool:
    """True iff this job has at least one IB (non-P2P) peer, i.e. the mesh spans NVLink domains.

    Semantics
        Delegates to the STORE's own connectivity probe
        (`fold_cp_ops.distributed.gemm_sm90_a2a.build_p2p_table`) and inverts it: the probe reports
        which peers are P2P-reachable, so "has IB peers" is "not all of them are". The import is
        function-local so that importing this module costs nothing on a rank that never asks.

    Args:
        pe_table: The flat-cp peer table -- any iterable of ints, e.g.
            ``pe_map.cp_pe_table.tolist()``. Its ORDER is not read here (only ``all()`` of the
            result), but it must be the cp table rather than an unordered roster: a table of the
            wrong PEs probes the wrong links and answers a different question. Must be non-empty;
            ``all(())`` is True, so an empty table reports "no IB peers" and would silently
            un-guard every caller.

    Returns:
        True when at least one entry of the probe is False.

    Raises:
        RuntimeError: From NVSHMEM, if called before initialization completes. Note this arrives as
            a hard process ABORT rather than a Python exception in some builds -- so call this only
            after ``DistributedManager.init_nvshmem()``, never at collection time.
    """
    from fold_cp_ops.distributed.gemm_sm90_a2a import build_p2p_table

    return not all(build_p2p_table(tuple(int(p) for p in pe_table)))


def skip_if_coupled_cross_node(pe_table, detail: str = "") -> None:
    """Skip a COUPLED (no ib_drain, no decoupled ring) peer-store test on a job with IB peers.

    Semantics
        Probes with :func:`job_has_ib_peers` and, if the mesh spans NVLink domains, skips through
        `rank_invariant_skip` rather than a bare ``pytest.skip``. Under ``tests/distributed/`` a
        skip whose predicate can differ per rank is a DEADLOCK rather than a skip, so the guard
        there requires the uniformity to be DECLARED; this one genuinely is uniform, and the
        declaration carries the upstream's own argument for why.

    Args:
        pe_table: The flat-cp peer table, as for :func:`job_has_ib_peers`.
        detail: Appended to the standard reason so a suite can name the alternative path that keeps
            the cross-node case covered. Optional; an empty string leaves the reason unchanged.

    Returns:
        None when the job is all-P2P, so the caller proceeds.

    Raises:
        Skipped: pytest's, when the job has IB peers.
    """
    if job_has_ib_peers(pe_table):
        rank_invariant_skip(
            COUPLED_NVLINK_ONLY_SKIP + (f" {detail}" if detail else ""),
            because=_UNIFORM_BECAUSE,
        )
