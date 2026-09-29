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

"""Tests for ``fold_cp_ops._internal.autotune.config`` -- one candidate, and its identity.

Everything else in the harness keys on `AutotuneConfig`: the timing dict, the disk cache, the
cross-rank tie-break. So the properties tested here are not about a data class -- they are what make
those three agree.

The identity has to be **type-distinguishing** and **stable across processes**. ``str()`` is neither:
``{'x': 1}`` and ``{'x': True}`` format identically and are different kernels, and a repr containing
an object address changes between runs, which silently voids the persistent cache.
"""


import pytest

from fold_cp_ops._internal.autotune import AutotuneConfig
from fold_cp_ops._internal.autotune.precompile import wire_admissible


def test_the_key_distinguishes_types_that_format_alike():
    """``1`` and ``True`` are different kernels, so they must be different configs.

    This is the concrete reason the key is built from ``(name, type name, value)`` triples rather
    than from a formatted string. Under a string key these two would collide, and the second one
    tuned would silently reuse the first one's compiled artifact and its timing.
    """
    assert AutotuneConfig(x=1) != AutotuneConfig(x=True)
    assert AutotuneConfig(x=1).key() != AutotuneConfig(x=True).key()
    assert len({AutotuneConfig(x=1), AutotuneConfig(x=True)}) == 2


def test_the_key_is_insensitive_to_keyword_order():
    """Two configs written in different orders are the same config.

    A cache keyed on insertion order would miss whenever a caller happened to list its knobs
    differently -- a re-tune rather than a wrong answer, but one that never converges.
    """
    a = AutotuneConfig(tile=128, cluster=2)
    b = AutotuneConfig(cluster=2, tile=128)
    assert a == b and a.key() == b.key() and hash(a) == hash(b)


def test_ordering_is_deterministic_across_equal_timings():
    """`__lt__` gives a total order, which is what makes a tie-break agree across ranks.

    Two ranks that measured two configs identically must still choose the SAME one. "Whichever the
    dict yielded first" is not that -- it depends on insertion order, which depends on the order
    `validity` happened to admit them.
    """
    pool = [AutotuneConfig(tile=t) for t in (256, 64, 128)]
    assert sorted(pool) == sorted(reversed(pool))
    assert min(pool) == min(reversed(pool))


def test_the_knobs_read_back_by_name_without_copying_the_pack():
    """``config[name]`` and ``config.get`` read one knob; ``all_kwargs`` hands back a fresh dict."""
    c = AutotuneConfig(tile=128, cluster=2)
    assert c["tile"] == 128
    assert c.get("cluster") == 2
    assert c.get("absent") is None
    assert c.get("absent", 7) == 7
    with pytest.raises(KeyError):
        c["absent"]
    kw = c.all_kwargs()
    kw["tile"] = 999
    assert c["tile"] == 128, "all_kwargs must return a copy, or a caller can mutate the config"


def test_a_config_survives_the_trip_to_a_precompile_worker():
    """The config's KNOBS cross the pipe and the config is REBUILT from them, not deserialized.

    The pre-compile workers are subprocesses, so every candidate crosses a pipe -- but what crosses
    is ``all_kwargs()``, a plain string-keyed mapping, and the worker constructs a fresh
    ``AutotuneConfig`` from it. That is the whole security property: the receiving side builds a
    known type from data, instead of letting the payload choose what to construct. It also means the
    key is rebuilt rather than trusted, so a corrupted or version-skewed key cannot be accepted as
    the identity of a config whose knobs say otherwise.

    This test used ``pickle.loads(pickle.dumps(c))``, which asserted a round trip the code no longer
    performs -- and would have kept passing after `pickle` was removed from the wire.
    """
    import msgpack

    c = AutotuneConfig(tile=128, pingpong=False, cluster=(1, 2))
    assert wire_admissible(c.all_kwargs()), "a shipped config must survive the worker protocol"
    on_wire = msgpack.unpackb(
        msgpack.packb(c.all_kwargs(), use_bin_type=True),
        raw=False,
        use_list=False,
        strict_map_key=True,
    )
    back = AutotuneConfig(**on_wire)
    assert back == c and back.key() == c.key()
    assert back.all_kwargs() == c.all_kwargs()
    assert back.all_kwargs()["cluster"] == (1, 2), "a tuple knob must not arrive as a list"


def test_the_repr_is_sorted_so_two_equal_configs_print_alike():
    """Equal configs must print identically, or a log line cannot be compared to another run's."""
    a, b = AutotuneConfig(tile=128, cluster=2), AutotuneConfig(cluster=2, tile=128)
    assert repr(a) == repr(b)
    assert "tile=128" in repr(a)


def test_every_identity_operation_goes_through_key_and_not_the_raw_slot():
    """`__eq__`, `__hash__` and `__lt__` must build the key, not read a slot that may be unset.

    The key is computed lazily, because construction sits on a per-launch path: every
    ``select="heuristic"`` front door builds one of these on every call just to read one value back
    out, and most are never keyed at all. That makes the raw ``_key`` slot ``None`` until something
    asks for it -- and the failure mode of reading it directly is total and silent. ``None == None``
    would report EVERY config equal to every other, underneath the result cache and the cross-rank
    tie-break, with nothing raising.

    So this pins the property rather than the implementation: distinct configs must be distinguished
    by each identity operation, on FRESH objects whose key nobody has requested yet.
    """
    a, b = AutotuneConfig(tile=128), AutotuneConfig(tile=256)
    assert a != b
    assert hash(AutotuneConfig(tile=128)) != hash(AutotuneConfig(tile=256))
    assert (AutotuneConfig(tile=128) < AutotuneConfig(tile=256)) != (
        AutotuneConfig(tile=256) < AutotuneConfig(tile=128)
    )
    assert AutotuneConfig(tile=128) == AutotuneConfig(tile=128)
    assert {AutotuneConfig(tile=128): "v"}[AutotuneConfig(tile=128)] == "v"


def test_a_runtime_value_is_refused_by_every_route_to_an_identity():
    """The admission test moved to `key()`, so it must bite on everything that needs a key.

    A config value is folded into the compiled kernel, hashed into the result cache and pickled to a
    pre-compile worker. A tensor corrupts all three, and none of them raises on its own. Running the
    test at first `key()` instead of at construction is safe ONLY because nothing can reach those
    three without an identity -- so the guarantee to pin is that every route to one refuses, not
    that the constructor does.
    """
    tensor_like = object()  # not an int/bool/str/float/None/tuple, so not a compile-time constant
    for route in (
        lambda c: c.key(),
        lambda c: hash(c),
        lambda c: c == AutotuneConfig(x=1),
        lambda c: c < AutotuneConfig(x=1),
    ):
        with pytest.raises(TypeError, match="compile-time constants"):
            route(AutotuneConfig(w=tensor_like))


def test_the_key_is_computed_once_and_reused():
    """Memoized, or the laziness would trade one per-call cost for a per-lookup one."""
    c = AutotuneConfig(tile=128, pingpong=True)
    assert c.key() is c.key()
