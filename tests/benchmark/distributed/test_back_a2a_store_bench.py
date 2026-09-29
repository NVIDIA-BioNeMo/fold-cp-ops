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

"""The back-A2A sweep dispatcher must never emit a kwarg the kernel no longer has.

Subject: ``benchmark/distributed/back_a2a_store_bench.py`` -- specifically ``_cluster_dispatch``
and ``_cluster_forced``, the two functions that turn a (variant, cluster_n, shape) point into the
``configure_a2a_gemm_native`` kwargs a sweep cell will pass.

Why this file exists (D0.1). Epic-close P2 made ``cluster_multislot`` the SOLE production IB drain and
REMOVED ``cluster_roundpark`` from ``configure_a2a_gemm_native``'s signature (``gemm_sm90_a2a.py``, the
"IB-drain variant is REMOVED at the source" block; the removal itself is pinned by
``tests/distributed/test_gemm_a2a_epi.py::test_removed_hybrid_drain_kwargs_rejected``, which asserts
``TypeError``). The dispatcher here was not updated with it, so every ``cluster_n >= 4`` STRADDLE cell
built an impossible kwargs dict and died with ``TypeError: ... unexpected keyword argument
'cluster_roundpark'`` -- measured 2026-08-20 as cluster_n=4: 8 of 12 valid, cluster_n=8: 0 ok / 288 err.

The tests are GPU-free by construction: both functions are pure and return
``(staging_shape, cfg_kwargs, label)``, so the defect is decidable without nvshmem, a process group or
a device. That matters -- the bug shipped precisely because reaching it needed a multi-rank sweep.
"""

from __future__ import annotations

import inspect

import pytest

import benchmark.distributed.back_a2a_store_bench as rp
from benchmark.distributed.harness.targets import cluster_drain as cd
from fold_cp_ops.distributed.gemm_sm90_a2a import GemmSm90A2A
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

pytestmark = matrix_exempt(
    "the subject is a pure host-side kwargs dispatcher -- there is no kernel, no operand and no "
    "shape axis to sweep; the parametrization below IS the whole input domain"
)

#: (cluster_n, nt_j_pp) covering every branch of ``_cluster_dispatch``: the degenerate cluster_n=1,
#: even-shard at each cluster_n, the cluster_n=2 straddle, and -- the regression -- both cluster_n>=4
#: straddles. cluster_n=8/nt_j_pp=12 is kept deliberately: it straddles AND satisfies the kernel's
#: ``cluster_n <= nt_j_pp`` multislot cap, so it is the case the fix must actually make runnable.
_DISPATCH_POINTS = [(1, 7), (1, 12), (2, 7), (2, 12), (4, 7), (4, 12), (4, 16), (8, 7), (8, 12), (8, 16)]


def _kernel_kwarg_names():
    """Every parameter name ``configure_a2a_gemm_native`` accepts, read from the live signature.

    Read rather than hardcoded: a hardcoded list would keep passing after the kernel dropped another
    kwarg, which is exactly the failure this file exists to catch.
    """
    sig = inspect.signature(GemmSm90A2A.configure_a2a_gemm_native)
    return {p for p in sig.parameters if p != "self"}


@pytest.mark.parametrize("cluster_n,nt_j_pp", _DISPATCH_POINTS)
def test_dispatch_emits_only_kwargs_the_kernel_accepts(cluster_n, nt_j_pp):
    """Every config ``_cluster_dispatch`` produces must be constructible by the kernel's front door.

    Checks the kwarg NAMES against the live signature rather than checking for the one known-bad name,
    so a future removal at the kernel source fails here instead of at rank 0 of a multi-node sweep.
    """
    _shape, kw, label = rp._cluster_dispatch(cluster_n, nt_j_pp, cp1=4, N_j_loc=1024, n_clusters=3, rd=2)
    unknown = set(kw) - _kernel_kwarg_names()
    assert not unknown, (
        f"_cluster_dispatch(cluster_n={cluster_n}, nt_j_pp={nt_j_pp}) -> label={label!r} emits "
        f"{sorted(unknown)}, which configure_a2a_gemm_native does not accept; that cell can only "
        f"raise TypeError at build time"
    )


@pytest.mark.parametrize("cluster_n,nt_j_pp", [(4, 7), (8, 7), (8, 12)])
def test_cluster_n_ge4_straddle_routes_to_the_surviving_drain(cluster_n, nt_j_pp):
    """The cn>=4 straddle region routes to ``cluster_multislot``, under a label of its own.

    The label must not be the historical ``roundpark_cp1buf``: a result row carrying that name would
    read as a round-park measurement, and round-park no longer exists.
    """
    assert nt_j_pp % cluster_n != 0, "the point must straddle or it tests the even-shard branch"
    _shape, kw, label = rp._cluster_dispatch(cluster_n, nt_j_pp, cp1=4, N_j_loc=1024, n_clusters=3, rd=2)
    assert kw == {"cluster_multislot": True}, f"expected the surviving drain, got {kw}"
    assert "roundpark" not in label, f"label {label!r} still claims a removed variant"


def test_forcing_the_removed_variant_raises_with_its_reason():
    """``_cluster_forced(force='roundpark')`` refuses, naming the removal.

    An old sbatch that still names the variant should learn WHY it cannot run. Before the fix it got
    a ``TypeError`` from three frames deeper, which says nothing about epic-close P2.
    """
    with pytest.raises(ValueError, match="REMOVED at the kernel source"):
        rp._cluster_forced("roundpark", 4, 7, cp1=4, N_j_loc=1024, n_clusters=3, rd=2)


def test_the_removed_variant_is_not_offered_anywhere():
    """Neither the sweep's variant list nor the harness adapter advertises the dead name."""
    assert "cluster_roundpark" not in rp.VARIANT_NAMES, rp.VARIANT_NAMES
    assert "cluster_roundpark" not in cd._FUSED, cd._FUSED


def test_the_kernel_really_has_no_such_parameter():
    """The premise of this whole file, asserted rather than assumed.

    If ``cluster_roundpark`` were ever restored to the kernel, these tests would be enforcing a
    restriction that no longer applies, and this is the one that would say so.
    """
    assert "cluster_roundpark" not in _kernel_kwarg_names()
