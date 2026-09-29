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

"""One tunable configuration, and the canonical identity every other layer keys on.

An `AutotuneConfig` is a frozen mapping of knob name to value -- ``tile_N=128, chunk_g=16`` -- that
the tuner passes to the kernel as keyword arguments. It looks trivial and it is the piece most of
the harness's correctness rests on, because FOUR different things need to agree on what "the same
config" means:

* the in-process cache, which maps a shape key to a winning config;
* the on-disk cache, which must survive a process restart and a different Python hash seed;
* the distributed consensus, where every rank must name the winner identically or the ranks pick
  different kernels and deadlock;
* the pre-compile worker pool, which receives configs over a pipe.

Upstream used ``str(config)`` as that identity, with a runtime assert that the strings happened to
be unique. That is the debt this module exists to retire: string formatting is not a contract, a
renamed knob silently invalidates every cached result, and two configs differing only in a value's
*type* (``1`` vs ``1.0``, ``True`` vs ``1``) format identically and collide.

:meth:`AutotuneConfig.key` is the identity instead -- a sorted tuple of ``(name, type name, value)``
triples. It is stable across processes, orderings and hash seeds, it distinguishes types, and it is
what the disk cache and the consensus both quote.

**Values must be compile-time constants.** The same admission test the kernel functors use
(`is_compile_time_value`) is applied here, because a config value is folded into a compiled kernel
exactly like a template parameter. A tensor in a config would be sent to a worker, hashed into a
cache key and baked into a kernel -- three different kinds of wrong, none of which raises on its own.
"""

from typing import Any, Dict, Tuple

from fold_cp_ops._internal.compile_time.template_params import is_compile_time_value


class AutotuneConfig:
    """A frozen set of tunable keyword arguments, with a canonical cross-process identity.

    Args:
        **kwargs: The knobs and their values. Every value must be a compile-time constant --
            ``int``, ``bool``, ``str``, ``float``, ``None``, a cutlass numeric TYPE, or a tuple of
            those. A tensor, layout or pointer raises: it cannot be folded into a kernel, cannot be
            hashed into a cache key, and cannot be sent to a pre-compile worker.

    Raises:
        TypeError: If any value is a runtime value. The message names the offending knobs and their
            types, because the alternative failure is silent -- a config holding a tensor produces a
            wrong cache key rather than an error. Raised on the first :meth:`key` call rather than at
            construction -- see the note on `__init__` below.
    """

    __slots__ = ("_kwargs", "_key")

    def __init__(self, **kwargs: Any):
        """Store the knobs. The admission test and the canonical key are DEFERRED to :meth:`key`.

        Purpose
            Construction is on a PER-LAUNCH path. Every ``select="heuristic"`` front door in this
            package builds one of these on every single call just to read one value back out of it,
            and most of those configs are never keyed at all -- nothing caches, serializes or compares
            a heuristic's answer, it is consumed and dropped.

        Semantics
            `_key` is left ``None`` and computed on first use, at which point the admission test
            runs too. Every path that actually CONSUMES an identity -- the result cache, the
            pre-compile worker, the distributed consensus -- goes through :meth:`key`, so each of
            them sees exactly the same validation and exactly the same tuple it saw before. What
            changes is only the cost of a config nobody keys.

            **This is a port-parity restoration, not an optimization.** ``main``'s counterpart
            (`_internal/autotuner.py`) is ``self.kwargs = kwargs`` and nothing else; the eager
            validation and eager key were ADDED here, and adding work to a per-call path is the same
            flavour of divergence as dropping it. Measured on the W1 front door's one-kwarg config:
            ours 1.306 us/call against main's 0.211, of which 0.270 is the admission test and 0.444
            the sorted-tuple build -- both now deferred.

        Args:
            **kwargs: The knobs and their values. Not validated here; see :meth:`key`. Storing an
                unvalidated value is safe precisely because it cannot reach a cache, a worker or a
                kernel without being keyed first.

        Returns:
            None.
        """
        object.__setattr__(self, "_kwargs", dict(kwargs))
        object.__setattr__(self, "_key", None)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse mutation: the identity is computed once, and a later edit would desynchronize it.

        Raises:
            AttributeError: Always. A config that changed after being cached would make the cache
                report a timing for a configuration that was never run.
        """
        raise AttributeError(
            f"AutotuneConfig is immutable ({name!r} cannot be set). Its key is computed at "
            f"construction and quoted by the result cache and the distributed consensus; a mutable "
            f"config would let those disagree about what was measured. Build a new one."
        )

    def key(self) -> Tuple[Tuple[str, str, Any], ...]:
        """The canonical identity: sorted ``(name, type name, value)`` triples.

        Stable across processes, insertion orders and hash seeds, and type-distinguishing -- which
        `str()` is not: ``{'x': 1}`` and ``{'x': True}`` format alike and are different kernels.

        Computed on FIRST call and memoized (see `__init__` for why it is not eager), which is also
        where the compile-time-constant admission test runs. Putting the test here rather than at
        construction is safe because a value cannot reach a compiled kernel, a result cache, a disk
        hash or a pre-compile worker without an identity, and this is the only way to get one --
        every consumer is downstream of this line.

        Returns:
            A tuple usable as a dict key, a sort key, and a component of a disk-cache hash. The same
            object on every subsequent call.

        Raises:
            TypeError: If any value is a runtime value (a tensor, layout or pointer). Names the
                offending knobs and their types.
        """
        key = self._key
        if key is None:  # `is None`, not falsy: a no-kwarg config's key is the empty tuple
            bad = {
                k: type(v).__name__ for k, v in self._kwargs.items() if not is_compile_time_value(v)
            }
            if bad:
                raise TypeError(
                    f"autotune config values must be compile-time constants; these are not: {bad}. "
                    f"A config value is folded into the compiled kernel, hashed into the result "
                    f"cache and sent to a pre-compile worker -- a runtime value silently "
                    f"corrupts all three. Pass it as a kernel argument instead."
                )
            key = tuple(sorted((k, type(v).__name__, v) for k, v in self._kwargs.items()))
            object.__setattr__(self, "_key", key)
        return key

    def all_kwargs(self) -> Dict[str, Any]:
        """The knobs, as keyword arguments for the kernel.

        Returns:
            A NEW dict each call, so a caller mutating the result cannot reach into the config.
        """
        return dict(self._kwargs)

    def get(self, name: str, default: Any = None) -> Any:
        """Read one knob without copying the whole pack.

        Purpose
            Validity and preference callbacks read one or two knobs per candidate and are called
            once per candidate per tuning run. ``all_kwargs()[name]`` allocates a dict to do that,
            and -- more to the point -- hands the caller a mutable copy of the pack when all it
            wanted was a value.

        Args:
            name: The knob's name, as declared by the axis or passed to the constructor.
            default: Returned when this config does not carry `name`. A default rather than a raise
                because a validity rule shared across two pools may legitimately ask about a knob
                only one of them declares.

        Returns:
            The knob's value, or `default`.
        """
        return self._kwargs.get(name, default)

    def __getitem__(self, name: str) -> Any:
        """The knob named `name`.

        Args:
            name: The knob's name.

        Returns:
            Its value.

        Raises:
            KeyError: If this config does not carry `name`. Use :meth:`get` for the tolerant form.
        """
        return self._kwargs[name]

    # These three go through `key()` and NEVER through `_key`. With the key computed lazily, the
    # raw slot is None until someone asks, and reading it directly here would be silent and total:
    # `__eq__` would compare None to None and report every config equal to every other, which the
    # result cache and the distributed consensus both build on.
    def __hash__(self) -> int:
        return hash(self.key())

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AutotuneConfig) and self.key() == other.key()

    def __lt__(self, other: "AutotuneConfig") -> bool:
        """Order by canonical key, so a tie-break between equal timings is deterministic.

        Load-bearing under a collective: two ranks that time two configs identically must still
        choose the SAME one, and "whichever the dict yielded first" is not that.
        """
        return [(k, t, str(v)) for k, t, v in self.key()] < [
            (k, t, str(v)) for k, t, v in other.key()
        ]

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(self._kwargs.items()))
        return f"AutotuneConfig({inner})"

    __str__ = __repr__

    def __getstate__(self) -> Dict[str, Any]:
        """Support pickling to a pre-compile worker; ``__slots__`` has no default ``__dict__``."""
        return dict(self._kwargs)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Rebuild through ``__init__`` so the key is recomputed rather than trusted from the wire."""
        AutotuneConfig.__init__(self, **state)
