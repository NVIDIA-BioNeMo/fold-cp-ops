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

"""Declaring the tunable axes, so they can be READ off the kernel rather than reverse-engineered.

A pool of `AutotuneConfig` points says *which combinations will be tried*. It does not say what the
axes are. Two very different declarations are indistinguishable once flattened::

    [AutotuneConfig(tM=m, tN=n)          {"tM", "tN"}, 8 opaque points
     for m in (64, 128)                  ── and ──
     for n in (32, 64, 128, 256)]        {"tM", "tN"}, 2 opaque points

    [AutotuneConfig(tM=64,  tN=32),      indistinguishable: is the second one a
     AutotuneConfig(tM=128, tN=256)]     sparse grid, or two hand-picked points?

`AxisSpace` declares the axes and GENERATES the points, so a reader (or a tool, or a doc generator)
can recover the axis names, their domains and the reason a grid is sparse. This mirrors
`fold_cp_ops.testing.kernel_matrix.Axis`, which already solves the same problem for the test matrix
-- and since the autotune pool and the test matrix are usually the *same knobs*, declaring both the
same way is what makes "the tests sweep tile_N to 256 but autotune only tries 128" a checkable
statement instead of a coincidence.

**There is no compile-vs-runtime annotation here, deliberately.** Which knobs are compile-time is not
a property of the tuning declaration -- it is a property of the kernel functor, and it is already
declared there: anything bound into `TemplateParams` is read off ``self`` during tracing and is
therefore folded into the kernel, while anything passed as an argument is not. Duplicating that here
would create a second source of truth that can disagree with the first. See
``docs/autotune_refactor.md`` §14.

The flat pool remains the underlying form -- `AxisSpace.configs()` simply produces it. That is what
keeps sparse and joint axes expressible: a combination that is illegal is excluded, not masked.
"""

import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from fold_cp_ops._internal.autotune.config import AutotuneConfig


class TuneAxis:
    """One tunable knob: its name, what the kernel claims to accept, and what will be tried.

    Args:
        name: The kernel keyword argument this axis sets. Must match a parameter of the decorated
            function, which `Autotuner` checks at decoration time.
        domain: Human-readable statement of what the kernel accepts along this axis -- *the claim*,
            not the pool. Never parsed; it exists so a reader can judge whether the pool is honest
            about the claim. Required, and required to be non-empty, for the same reason
            `kernel_matrix.Axis` requires it: an axis whose domain nobody wrote down is one nobody
            can tell is under-covered.
        values: The values to try. Must be non-empty and free of duplicates -- a duplicate would be
            benchmarked twice and could win against itself.

    Raises:
        ValueError: If `values` is empty or contains duplicates, or if `domain` is blank.
    """

    __slots__ = ("name", "domain", "values")

    def __init__(self, name: str, domain: str, values: Sequence[Any]):
        if not domain or not domain.strip():
            raise ValueError(
                f"TuneAxis({name!r}) needs a non-empty `domain` stating what the kernel ACCEPTS "
                f"along this axis. Without it there is no way to tell a deliberately narrow pool "
                f"from a forgotten one."
            )
        values = tuple(values)
        if not values:
            raise ValueError(f"TuneAxis({name!r}) has no values to try")
        if len(set(map(repr, values))) != len(values):
            raise ValueError(f"TuneAxis({name!r}) has duplicate values: {values}")
        self.name = name
        self.domain = domain
        self.values = values

    def __repr__(self) -> str:
        return f"TuneAxis({self.name!r}, values={self.values!r})"


class AxisSpace:
    """A declared set of axes, and the config pool their product generates.

    Args:
        *axes: The `TuneAxis` declarations. At least one; names must be unique.
        exclude: Optional ``(dict) -> bool`` receiving one point as ``{name: value}``. Returning
            True drops that point. This is how a SPARSE grid stays readable: the exclusion is a
            written rule rather than an unexplained list of survivors. It must be a pure function of
            the point -- it may not consult the request, the device or the environment, because the
            pool must be identical on every rank (see `ConfigSpace`).
        because: Why the exclusion exists. **Required whenever `exclude` is given** -- an
            unexplained hole in a grid is indistinguishable from a forgotten combination.

    Raises:
        ValueError: If no axes are given, if two axes share a name, if `exclude` is given without
            `because`, or if the exclusion removes every point.
    """

    def __init__(
        self,
        *axes: TuneAxis,
        exclude: Optional[Callable[[Dict[str, Any]], bool]] = None,
        because: Optional[str] = None,
    ):
        if not axes:
            raise ValueError("an AxisSpace needs at least one TuneAxis")
        names = [a.name for a in axes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate axis name in {names}")
        if exclude is not None and not (because and because.strip()):
            raise ValueError(
                "AxisSpace(exclude=...) requires `because=`: a hole in the grid must say why it is "
                "there, or it cannot be told from a combination nobody considered."
            )
        self.axes: Tuple[TuneAxis, ...] = tuple(axes)
        self._exclude = exclude
        self.because = because
        if not self.configs():
            raise ValueError(
                f"the exclusion removed every point from the grid (because={because!r}). "
                f"That leaves the kernel with nothing to run."
            )

    def points(self) -> Iterable[Dict[str, Any]]:
        """Every surviving point of the grid, as ``{axis name: value}`` dicts.

        Returns:
            An iterator in declaration order -- axis 0 varying slowest -- so the generated pool is
            deterministic and therefore identical on every rank.
        """
        for combo in itertools.product(*(a.values for a in self.axes)):
            point = dict(zip((a.name for a in self.axes), combo))
            if self._exclude is None or not self._exclude(point):
                yield point

    def configs(self) -> List[AutotuneConfig]:
        """The generated config pool -- the flat form every other layer consumes.

        Returns:
            One `AutotuneConfig` per surviving point, in declaration order.
        """
        return [AutotuneConfig(**p) for p in self.points()]

    def excluded_count(self) -> int:
        """How many points of the full product the exclusion removed.

        Reported rather than merely applied: a sparse grid should be visibly sparse. Returns 0 when
        there is no exclusion.
        """
        full = 1
        for a in self.axes:
            full *= len(a.values)
        return full - len(self.configs())

    def describe(self) -> str:
        """A readable summary of the declared space, for logs and diagnostics.

        Returns:
            One line per axis with its domain and values, plus the exclusion and its reason. This is
            the thing that was impossible to produce from a flat pool.
        """
        lines = [f"{len(self.configs())} configs over {len(self.axes)} axes:"]
        for a in self.axes:
            lines.append(f"  {a.name}: {list(a.values)}   -- {a.domain}")
        if self._exclude is not None:
            lines.append(f"  excluded {self.excluded_count()} point(s): {self.because}")
        return "\n".join(lines)
