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

"""Tests for ``fold_cp_ops._internal.autotune.space`` -- the candidate pool.

The whole module exists to keep two things apart that upstream ran together in one
``prune_configs_by`` callback:

* **validity** MAY REJECT. A rejected config cannot run at all -- wrong alignment, a schedule the
  request forbids.
* **preference** MAY ONLY REORDER. It exists so `top_k` keeps the promising configs, and it must
  never remove one.

Collapsing them makes "we did not measure this config" and "this config cannot run" the same event,
which is how a pool silently narrows. The tests below are mostly about that separation, plus the one
property the distributed case rests on: `candidates()` must be a deterministic function of the
request, because every rank computes it independently and they have to agree.
"""

import pytest

from fold_cp_ops._internal.autotune import AutotuneConfig
from fold_cp_ops._internal.autotune.space import ConfigSpace

_POOL = [AutotuneConfig(tile=t) for t in (64, 128, 256)]


def test_an_empty_pool_is_refused_at_construction():
    """A pool with nothing in it leaves the kernel with nothing to run, and no way to say so."""
    with pytest.raises(ValueError, match=r"at least one config"):
        ConfigSpace([])


def test_a_duplicate_config_is_refused_naming_it():
    """A duplicate would be benchmarked twice and could win against itself.

    It also makes the on-disk result map ambiguous: two entries for one key, and whichever is read
    last decides the winner. Refusing at construction is the only place this is cheap to catch.
    """
    with pytest.raises(ValueError, match=r"duplicate configs"):
        ConfigSpace([AutotuneConfig(tile=64), AutotuneConfig(tile=64)])


@pytest.mark.parametrize("bad", [0, -1])
def test_a_top_k_below_one_is_refused(bad):
    """A cap of zero would truncate the pool to nothing, which is the empty-pool failure again."""
    with pytest.raises(ValueError, match=r"top_k must be >= 1"):
        ConfigSpace(_POOL, top_k=bad)


def test_validity_rejects_and_the_survivors_keep_declaration_order():
    """Rejection is by config, and the order the author declared is the order that survives.

    Order matters beyond aesthetics: with ``CPO_AUTOTUNE=0`` the tuner runs ``candidates()[0]``, so
    declaration order IS the documented default. A validity rule that reordered would change the
    untuned kernel.
    """
    space = ConfigSpace(_POOL, validity=lambda c, r: c["tile"] <= r["max_tile"])
    assert [c["tile"] for c in space.admissible({"max_tile": 128})] == [64, 128]
    assert [c["tile"] for c in space.candidates({"max_tile": 256})] == [64, 128, 256]


def test_rejecting_everything_is_a_loud_error_not_an_empty_list():
    """No legal configuration for a shape is a real condition and must fail where it happens.

    Returning an empty list instead surfaces several frames later as ``min() of empty sequence``,
    which names neither the kernel nor the shape.
    """
    space = ConfigSpace(_POOL, validity=lambda c, r: False)
    with pytest.raises(ValueError, match=r"rejected as INVALID"):
        space.admissible({"shape": (1, 2)})


def test_preference_reorders_and_top_k_truncates_what_preference_ordered():
    """`prefer` decides WHICH configs `top_k` keeps -- that is the only reason it exists."""
    space = ConfigSpace(_POOL, prefer=lambda c, r: -c["tile"], top_k=2)
    assert [c["tile"] for c in space.candidates({})] == [256, 128]
    assert space.truncated({}) == 1


def test_preference_ties_are_broken_by_the_canonical_key_not_by_pool_order():
    """Equal preference scores must still give a deterministic order.

    Every rank computes `candidates()` independently, so a list whose order depended on the pool's
    declaration order would drift the moment someone reordered the declaration -- and two ranks on
    different commits would then benchmark different sequences and compare the results.
    """
    space = ConfigSpace(_POOL, prefer=lambda c, r: 0)
    forward = [c["tile"] for c in space.candidates({})]
    reversed_space = ConfigSpace(list(reversed(_POOL)), prefer=lambda c, r: 0)
    assert forward == [c["tile"] for c in reversed_space.candidates({})]


def test_truncated_reports_zero_when_nothing_was_capped():
    """`truncated` is for reporting, so it must be silent when there is nothing to report.

    A bounded sweep that does not say what it dropped reads as "we measured everything" -- which is
    the failure CLAUDE.md's no-silent-caps rule names. Reporting a spurious non-zero would be the
    same problem from the other side.
    """
    assert ConfigSpace(_POOL).truncated({}) == 0
    assert ConfigSpace(_POOL, top_k=3).truncated({}) == 0
    assert ConfigSpace(_POOL, top_k=1).truncated({}) == 2


def test_candidates_is_deterministic_for_one_request():
    """Two independent evaluations give the identical list -- the property consensus rests on.

    Ranks do not exchange their candidate lists; they each compute one and assume the others match.
    If they did not, the max-reduced timings would be reducing measurements of different kernels.
    """
    space = ConfigSpace(
        _POOL, validity=lambda c, r: c["tile"] != 128, prefer=lambda c, r: c["tile"]
    )
    request = {"m": 4096}
    assert space.candidates(request) == space.candidates(request)
