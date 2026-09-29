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

"""Toggle a codegen-affecting class attribute in a test without measuring the wrong artifact.

**The hazard this exists to remove, stated as the measurement that found it.** A kernel functor's
class attributes can change what the compiler emits -- ``GemmSm90._KEEP_STATIC_LEN_K`` decides
whether the producer's k-loop is a counted loop or a top-tested one with a ``BREAK``. None of them
are arguments to the ``@jit_cache``-decorated compile entry, whose key is
``(fn.__qualname__, *args, **sorted(kwargs))``. So two different values of such an attribute hash to
the SAME key. Measured, with a fresh ``CPO_CACHE_DIR`` and the cache enabled: compiling one shape at
``_KEEP_STATIC_LEN_K = True`` produced 4 ``.o`` files, and re-running the identical shape at
``False`` produced **4** -- no new artifact. The second setting silently ran the first one's
compiled code, and a test comparing them would have reported "identical behaviour" while never
having compiled the second variant at all.

**Why this is a test hazard and not a production one.** In production such an attribute is fixed by
the class, and the class is selected by an argument that IS in the key (``fusion_variant`` picks the
functor). So the key is complete over every state production can reach. It is incomplete only under
*runtime mutation of the attribute*, which nothing but a test does. Editing the ``.py`` to change a
default is also safe: ``cache_utils._compute_source_fingerprint()`` hashes every ``.py`` under the
package root, so a source edit lands in a different cache directory.

**Why the fix is a helper and not a cache-key change.** Adding the attribute to the key means
threading a parameter through the compile signature that no production caller ever varies -- one
more thing to keep in sync, bought for tests alone. And it would treat one flag as special when the
hazard belongs to the whole class: ``_STAGES_LN_AFFINE`` and ``_SUPPORTS_PINGPONG`` have exactly the
same shape and are safe today only because nothing mutates them. This module makes the safe way the
only convenient way, for any such attribute.

Both guards below are necessary and neither is sufficient:

* the **disk** cache is keyed by the source fingerprint, which a runtime toggle does not change, so
  it must be off (``CPO_CACHE_ENABLED=0``);
* the **in-process** memo is a plain dict that survives inside one interpreter regardless of the
  disk setting, so it must be cleared on the way in AND on the way out -- otherwise the toggle
  leaks its artifacts to whatever runs next in the same process.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any, Iterator

__all__ = ["clear_jit_caches", "codegen_flag", "iter_jit_caches"]

#: Attributes `cache_utils.jit_cache` attaches to its wrapper. Used to recognise a decorated
#: function without importing it, so this module stays independent of which entries exist.
_JIT_CACHE_MARKERS = ("cache", "cache_clear", "cache_info")


def iter_jit_caches() -> Iterator[Any]:
    """Yield every ``@jit_cache``-decorated wrapper reachable from an imported ``fold_cp_ops`` module.

    Semantics:
        Walks ``sys.modules`` rather than a hand-maintained registry, so an entry added later is
        picked up with no change here. Recognition is structural -- an object carrying all of
        ``cache``, ``cache_clear`` and ``cache_info`` -- because importing the decorator to compare
        identities would not help: ``functools.wraps`` hides it. Duplicates are suppressed by
        ``id``, since one wrapper is commonly re-exported from several modules and clearing it twice
        is merely wasteful, but yielding it twice would make a caller's count meaningless.

        Only IMPORTED modules are visited. A compile entry in a module nothing has imported has no
        populated memo, so it cannot serve a stale artifact.

    Returns:
        An iterator of wrapper objects, each exposing ``cache_clear()``. Order is unspecified.
    """
    seen: set[int] = set()
    for name, mod in list(sys.modules.items()):
        if not name.startswith("fold_cp_ops") or mod is None:
            continue
        for obj in list(vars(mod).values()):
            if not callable(obj) or id(obj) in seen:
                continue
            if all(hasattr(obj, m) for m in _JIT_CACHE_MARKERS):
                seen.add(id(obj))
                yield obj


def clear_jit_caches() -> int:
    """Empty every in-process ``@jit_cache`` memo.

    Semantics:
        Calls ``cache_clear()`` on each wrapper :func:`iter_jit_caches` finds, which drops the
        memo dict and resets its hit/miss counters. It does NOT touch the on-disk ``.o`` cache --
        that is keyed by the source fingerprint and is the caller's job to disable, which
        :func:`codegen_flag` enforces.

    Returns:
        The number of memos cleared. Zero means nothing was imported yet, which is fine before the
        first compile and suspicious after one -- a caller that expects to have invalidated
        something can assert on it.
    """
    n = 0
    for wrapper in iter_jit_caches():
        wrapper.cache_clear()
        n += 1
    return n


@contextlib.contextmanager
def codegen_flag(owner: type, name: str, value: Any) -> Iterator[None]:
    """Set a codegen-affecting class attribute for the duration of the block, cache-safely.

    Use this instead of assigning the attribute directly. A bare assignment compiles nothing new:
    the ``@jit_cache`` key does not include the attribute, so the next call is served the artifact
    built under the previous value and the test passes while measuring the wrong code.

    Semantics:
        On entry: verifies the disk cache is off, clears every in-process memo, then sets the
        attribute. On exit: restores the previous state and clears the memos again, so the artifacts
        compiled under the temporary value cannot leak to later tests in the same process. The
        restore is in a ``finally``, so it holds even if the body raises.

        Restoring is done by rebinding or by ``delattr``, whichever returns ``owner`` to its
        original state -- an attribute INHERITED from a base class must be deleted rather than
        rewritten, or the subclass would keep a shadowing copy that happens to equal the base's
        value today and silently stops tracking it if the base changes.

    Args:
        owner: The class to mutate. Must be the class that should carry the attribute during the
            block; passing a BASE class when a subclass overrides the attribute changes nothing
            observable, because the subclass's own binding still wins.
        name: Attribute name, e.g. ``"_KEEP_STATIC_LEN_K"``. Must already exist on ``owner`` or be
            inherited by it -- a typo would otherwise install a new attribute that nothing reads and
            the test would silently measure the untoggled default.
        value: The temporary value. No type check: what is valid depends on the attribute.

    Yields:
        None. The attribute is in effect for the body only.

    Raises:
        RuntimeError: If the persistent ``.o`` cache is enabled. It is refused rather than disabled
            here on purpose -- ``cache_utils.CACHE_ENABLED`` is read at import time into a module
            global, so flipping it now would not affect an already-imported reader, and a helper
            that appeared to fix the problem without fixing it is worse than one that stops.
        AttributeError: If ``name`` is not present on ``owner`` or any of its bases.
    """
    from fold_cp_ops._internal import cache_utils

    if cache_utils.CACHE_ENABLED:
        raise RuntimeError(
            f"codegen_flag({owner.__name__}, {name!r}) needs the persistent .o cache OFF: the "
            "attribute is not part of the jit_cache key, so with the cache on the toggled value is "
            "served the untoggled artifact and the test measures nothing. Set CPO_CACHE_ENABLED=0 "
            "in the environment BEFORE importing fold_cp_ops (it is read once at import)."
        )
    if not any(name in vars(klass) for klass in owner.__mro__):
        raise AttributeError(
            f"{owner.__name__} has no attribute {name!r} on itself or any base; refusing to create "
            "one, because a typo here reads as 'the flag had no effect' rather than as an error."
        )

    had_own = name in vars(owner)
    previous = vars(owner).get(name)
    clear_jit_caches()
    setattr(owner, name, value)
    try:
        yield
    finally:
        if had_own:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)
        clear_jit_caches()
