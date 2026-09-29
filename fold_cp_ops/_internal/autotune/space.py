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

"""The config space, split into a VALIDITY layer and a PERF layer -- and the split is enforced.

This is the single most consequential structural change relative to the upstream harness, and the
reason is a defect class rather than a preference.

Upstream had one hook, ``early_config_prune``, doing two unrelated jobs:

* deciding a config **cannot run** on this shape (a tile wider than the tensor, a `chunk_g` the
  alignment forbids) -- an arch-INDEPENDENT correctness fact; and
* deciding a config is **probably slow** -- an arch-DEPENDENT performance guess, bisected on one
  specific part.

Merging them means a perf guess can silently remove a config that was the only *legal* one, and a
validity rule can be mistaken for a tuning opinion and "relaxed" by someone chasing a number. It
also makes the whole prune arch-dependent, so moving to a different GPU can change which configs are
considered legal.

Here they are separate arguments with different powers:

* ``validity`` may REJECT. It must be a pure function of the request (shapes, dtypes, layouts) and
  must not consult the device -- a config that cannot launch cannot launch on any part.
* ``prefer`` may only REORDER, and ``top_k`` may only TRUNCATE what `prefer` ordered. Neither can
  remove a config that `validity` admitted... except through `top_k`, which is why `top_k` refuses
  to cut the list below one and why truncation is reported rather than silent.

**Why this matters more under a collective.** Every rank must arrive at the SAME candidate list, or
they benchmark different things and the consensus compares timings that are not comparable. A
`validity` that is a pure function of the request is identical on every rank by construction. One
that peeked at the device -- free memory, clock, an env var -- would not be, and the divergence
would surface as a hang, not as an error. That requirement is stated here because it cannot be
checked: this module cannot prove a callable is pure.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

from fold_cp_ops._internal.autotune.config import AutotuneConfig


class ConfigSpace:
    """The candidate configs for one kernel, with validity and preference kept apart.

    Args:
        configs: The declared pool. Must be non-empty and free of duplicate keys -- a duplicate
            would be benchmarked twice and could win against itself, and it makes the on-disk
            result map ambiguous.
        validity: Optional ``(config, request) -> bool``. Returning False means the config CANNOT
            run for this request. Must be a pure function of the request: no device queries, no
            env vars, no globals. A non-pure one makes ranks disagree about the candidate list,
            which under a collective is a hang rather than a wrong answer.
        prefer: Optional ``(config, request) -> sort key``. Orders admissible configs cheapest-first
            so that `top_k` keeps the promising ones. May NOT reject; its return value is only ever
            used to sort.
        top_k: Optional cap on how many admissible configs are actually timed. ``None`` times all
            of them. Must be >= 1 -- a cap of zero would leave nothing to run.

    Raises:
        ValueError: If `configs` is empty, contains duplicates, or `top_k` is less than 1.
    """

    def __init__(
        self,
        configs: Sequence[AutotuneConfig],
        *,
        validity: Optional[Callable[[AutotuneConfig, Dict[str, Any]], bool]] = None,
        prefer: Optional[Callable[[AutotuneConfig, Dict[str, Any]], Any]] = None,
        top_k: Optional[int] = None,
    ):
        configs = list(configs)
        if not configs:
            raise ValueError(
                "a ConfigSpace needs at least one config. An empty pool would leave the kernel "
                "with nothing to run and no way to say so."
            )
        keys = [c.key() for c in configs]
        if len(set(keys)) != len(keys):
            dupes = sorted({str(c) for c in configs if keys.count(c.key()) > 1})
            raise ValueError(
                f"duplicate configs in the pool: {dupes}. A duplicate is benchmarked twice, can "
                f"win against itself, and makes the on-disk result map ambiguous."
            )
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None; got {top_k}")
        self.configs = configs
        self._validity = validity
        self._prefer = prefer
        self.top_k = top_k

    def admissible(self, request: Dict[str, Any]) -> List[AutotuneConfig]:
        """The configs that CAN run for this request, in declaration order.

        Args:
            request: The call's shape/dtype/layout facts, as a plain dict. Passed to `validity`
                unchanged.

        Returns:
            The admitted configs. Never empty -- see Raises.

        Raises:
            ValueError: If `validity` rejected every config. That is a real condition worth a loud
                failure: it means the kernel has no legal configuration for this shape, and the
                alternative (returning an empty list) surfaces much later as an unhelpful
                "min() of empty sequence".
        """
        if self._validity is None:
            return list(self.configs)
        kept = [c for c in self.configs if self._validity(c, request)]
        if not kept:
            raise ValueError(
                f"every config in the pool was rejected as INVALID for this request "
                f"({ {k: v for k, v in request.items() if not k.startswith('_')} }). The kernel "
                f"has no legal configuration for this shape -- widen the pool, or relax the "
                f"validity rule if it is stricter than the hardware."
            )
        return kept

    def candidates(self, request: Dict[str, Any]) -> List[AutotuneConfig]:
        """The configs to actually benchmark: admissible, preference-ordered, `top_k`-truncated.

        Args:
            request: The call's shape/dtype/layout facts.

        Returns:
            At least one config. Deterministic for a given request -- ties in `prefer` are broken
            by the configs' canonical keys, so two ranks computing this independently get
            identical lists, which is what makes the distributed consensus meaningful.
        """
        kept = self.admissible(request)
        if self._prefer is not None:
            # The canonical key is the secondary sort so that equal preference scores do not leave
            # the order up to the pool's declaration order, which a caller may reasonably change.
            kept = sorted(kept, key=lambda c: (self._prefer(c, request), c.key()))
        if self.top_k is not None:
            kept = kept[: self.top_k]
        return kept

    def truncated(self, request: Dict[str, Any]) -> int:
        """How many admissible configs `top_k` dropped, for reporting rather than for control flow.

        A silently truncated sweep reads as "we measured everything"; this is what lets the tuner
        say otherwise. See CLAUDE.md's rule that a bounded sweep must log what it dropped.

        Args:
            request: The call's shape/dtype/layout facts.

        Returns:
            The count of admissible configs not benchmarked. 0 when `top_k` is None.
        """
        if self.top_k is None:
            return 0
        return max(0, len(self.admissible(request)) - self.top_k)
