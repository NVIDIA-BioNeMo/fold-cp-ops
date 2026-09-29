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

"""Tests for ``fold_cp_ops._internal.autotune.cache`` -- the on-disk record of a sweep.

**Every failure mode here is silent**, which is why this file is mostly about what the cache
REFUSES to return rather than about what it stores:

* A hit that should have been a miss reuses a timing measured for a different candidate set, a
  different device, or a different shape. Nothing raises; the kernel simply runs the wrong config.
* A partial hit, if merged, compares a fresh timing against a stale one measured under different
  thermal conditions -- a comparison with no meaning that still produces a winner.
* A torn write, if not atomic, is a JSON parse error at best and a truncated timing map at worst.

The cache is off by default for the same reason: a stale timing is a silent perf regression, and
opting in should be a decision a deployment makes once rather than a default nobody noticed.
"""

import json
import os
import re
import stat

import pytest

from fold_cp_ops._internal import cache_security
from fold_cp_ops._internal.autotune import AutotuneConfig
from fold_cp_ops._internal.autotune.cache import ResultCache, cache_root

_A, _B = AutotuneConfig(tile=64), AutotuneConfig(tile=128)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An enabled `ResultCache` writing under a temporary directory.

    Pins ``CPO_AUTOTUNE_CACHE_DIR`` so nothing touches a developer's real cache, and clears
    ``CPO_AUTOTUNE_REFRESH`` so an exported one in the ambient shell cannot make every load a miss
    and every assertion below vacuous.

    Returns:
        A `ResultCache` for a kernel named ``"k"``, with `enabled=True`.
    """
    monkeypatch.setenv("CPO_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CPO_AUTOTUNE_REFRESH", raising=False)
    return ResultCache("k", enabled=True)


def test_the_cache_is_on_by_default_and_the_env_var_can_disable_it(monkeypatch):
    """Default-ON, matching `main`, with the env var able to turn it OFF.

    This assertion is inverted from what it once was, and the inversion is the point. `main`'s
    ``autotune`` decorator defaults ``cache_results=True``, so `main` sweeps a shape ONCE EVER and
    every later process loads the same winner. Defaulting off here made this package re-sweep in
    every process, and a candidate that is genuinely BIMODAL then gets elected whenever a sweep
    catches its fast sample -- measured on op 7 at N_token=2048/D=256, tile (128,160) timed
    5.955 / 11.249 / 6.411 ms in one process and cost ~16% end-to-end when it won. Non-determinism
    in config selection is a worse failure than the stale timing the old default guarded against,
    and staleness is already handled by keying on the candidate set (see `_path`).
    """
    monkeypatch.delenv("CPO_AUTOTUNE_CACHE", raising=False)
    assert ResultCache("k").enabled is True
    monkeypatch.setenv("CPO_AUTOTUNE_CACHE", "0")
    assert ResultCache("k").enabled is False
    monkeypatch.setenv("CPO_AUTOTUNE_CACHE", "1")
    assert ResultCache("k").enabled is True
    assert ResultCache("k", enabled=False).enabled is False, "the explicit flag must win"


def test_a_stored_sweep_reads_back_as_the_same_configs(cache):
    """Round-trip: the keys come back as config OBJECTS, not as the strings they were stored as."""
    cache.store(("m", 4096), {_A: 2.0, _B: 1.0})
    got = cache.load(("m", 4096), [_A, _B])
    assert got == {_A: 2.0, _B: 1.0}
    assert all(isinstance(c, AutotuneConfig) for c in got)


def test_a_different_shape_is_a_miss(cache):
    """The shape is in the key, so a new shape must re-measure rather than inherit a winner."""
    cache.store(("m", 4096), {_A: 2.0, _B: 1.0})
    assert cache.load(("m", 8192), [_A, _B]) is None


def test_adding_a_config_to_the_pool_invalidates_the_entry(cache):
    """The CANDIDATE SET is in the key, and this is the subtle one.

    If it were not, adding a config to the pool would return the old winner from cache and the new
    config would never be measured -- which reads exactly like "the new config is not faster". The
    developer who added it has no way to tell the difference.
    """
    cache.store(("m", 4096), {_A: 2.0})
    assert cache.load(("m", 4096), [_A]) == {_A: 2.0}
    assert cache.load(("m", 4096), [_A, _B]) is None, (
        "a wider candidate set must MISS; returning the narrow sweep's winner would hide the new "
        "config entirely"
    )


def test_a_partial_hit_is_a_miss_rather_than_a_merge(cache):
    """Never mix a fresh timing with a stored one.

    They were measured under different thermal conditions, on different clock states, possibly
    minutes apart. Comparing them still produces a winner, and the winner means nothing.
    """
    cache.store(("m", 4096), {_A: 2.0, _B: 1.0})
    assert cache.load(("m", 4096), [_A]) is None


def test_refresh_forces_a_re_measure(cache, monkeypatch):
    """``CPO_AUTOTUNE_REFRESH=1`` makes every load a miss, without deleting anything.

    That is the escape hatch for "the numbers look wrong": re-measure now, keep the old file for
    comparison, and turn the flag off again.
    """
    cache.store(("m", 4096), {_A: 2.0, _B: 1.0})
    monkeypatch.setenv("CPO_AUTOTUNE_REFRESH", "1")
    assert cache.load(("m", 4096), [_A, _B]) is None
    monkeypatch.setenv("CPO_AUTOTUNE_REFRESH", "0")
    assert cache.load(("m", 4096), [_A, _B]) is not None


def test_a_corrupt_file_is_a_miss_not_a_crash(cache, tmp_path):
    """An unreadable entry re-measures. A cache is an optimization; it may not break a run."""
    cache.store(("m", 4096), {_A: 1.0})
    written = list(tmp_path.glob("k.*.json"))
    assert written, "nothing was written, so this test is vacuous"
    written[0].write_text("{not json")
    assert cache.load(("m", 4096), [_A]) is None


def test_a_non_writer_rank_does_not_write(cache, tmp_path):
    """Under a collective only rank 0 writes; every rank holds the same reconciled result.

    Having all of them write is redundant and, on a shared filesystem, risks a torn read even with
    an atomic replace per writer -- N writers replacing the same path is N chances to interleave.
    """
    cache.store(("m", 4096), {_A: 1.0}, is_writer=False)
    assert not list(tmp_path.glob("k.*.json"))
    cache.store(("m", 4096), {_A: 1.0}, is_writer=True)
    assert list(tmp_path.glob("k.*.json"))


def test_an_empty_sweep_is_not_recorded(cache, tmp_path):
    """Storing nothing writes nothing, so a failed sweep cannot poison the next run with a hit."""
    cache.store(("m", 4096), {})
    assert not list(tmp_path.glob("k.*.json"))


def test_the_device_is_part_of_the_file_identity(cache, tmp_path):
    """The stored payload names the device, so a file from another GPU is legible as such.

    The device is also hashed into the FILENAME -- that is what makes a shared cache directory safe
    across a heterogeneous cluster. This asserts the readable half, since the hashed half cannot be
    observed without a second GPU model.
    """
    cache.store(("m", 4096), {_A: 1.0})
    payload = json.loads(list(tmp_path.glob("k.*.json"))[0].read_text())
    assert payload["device"]
    assert payload["kernel"] == "k"
    assert "readable" in payload, "a human must be able to read which config won without decoding"


def test_the_cache_root_follows_the_environment(tmp_path, monkeypatch):
    """``CPO_AUTOTUNE_CACHE_DIR`` wins, and the directory is created rather than assumed."""
    target = tmp_path / "nested" / "dir"
    monkeypatch.setenv("CPO_AUTOTUNE_CACHE_DIR", str(target))
    assert cache_root() == target
    assert target.is_dir()


def test_describe_env_lists_every_variable_the_module_reads():
    """The help text and the code must not drift; an undocumented flag is an unusable one.

    The second half asserts that every variable the help text NAMES carries this project's own
    prefix. It replaces a check that read ``os.environ`` and so tested the ambient shell rather than
    the module: it could only fail when a foreign-prefixed variable happened to be exported in the
    running process AND was also named by ``describe_env()``, which is to say essentially never.
    Reading the text itself is what makes the assertion about the code under test.
    """
    text = "\n".join(ResultCache.describe_env())
    for var in ("CPO_AUTOTUNE_CACHE", "CPO_AUTOTUNE_CACHE_DIR", "CPO_AUTOTUNE_REFRESH"):
        assert var in text, f"{var} is read by the module but absent from describe_env()"
    named = set(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", text))
    assert named, "describe_env() named no variables at all; the pattern or the help text changed"
    foreign = sorted(v for v in named if not v.startswith("CPO_"))
    assert not foreign, f"describe_env() names non-CPO_ variables: {foreign}"


def test_the_cache_root_is_created_0700_and_the_stored_json_is_0600(cache, tmp_path):
    """Timings are not code, but a hostile entry here still mis-steers a kernel choice.

    A config picked from somebody else's numbers is a wrong answer about performance with nothing
    pointing at its cause -- the run is correct, merely slow, and the sweep that would have caught
    it is the very thing being short-circuited. Cheap to close, so it is closed.
    """
    cache.store(("m", 4096), {_A: 1.0})
    entries = list(tmp_path.glob("k.*.json"))
    assert entries, "nothing was stored, so this test proved nothing"
    assert stat.S_IMODE(os.stat(tmp_path).st_mode) == 0o700, (
        f"cache root is {stat.S_IMODE(os.stat(tmp_path).st_mode):04o}, expected 0700"
    )
    for path in entries:
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600, (
            f"{path.name} is {stat.S_IMODE(os.stat(path).st_mode):04o}, expected 0600"
        )


def test_the_stored_payload_keys_are_exactly_the_declared_six(cache, tmp_path):
    """Pin the schema, because this file is written by one process and read by another.

    `ResultCache.store` records request/configuration metadata and timings -- never tensor contents
    or weights -- and that is a release claim, not an implementation detail. An extra key added
    later would slip past every other test in this file, all of which look up keys they expect
    rather than enumerate what is there.
    """
    cache.store(("m", 4096), {_A: 1.0, _B: 2.0})
    payload = json.loads(list(tmp_path.glob("k.*.json"))[0].read_text())
    assert set(payload) == {"schema", "kernel", "key", "device", "timings", "readable"}, (
        f"the stored schema changed: {sorted(payload)}"
    )


def test_an_unsafe_root_makes_load_a_miss_and_store_a_no_op(tmp_path, monkeypatch):
    """A refused root must degrade to a cold cache, never to an exception.

    Both halves are asserted. A `store` that raised would fail a tuning run outright; a `load` that
    raised would fail it on the next process. The correct behaviour for both is silence plus a
    re-measurement, which costs exactly what having no cache costs.
    """
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    os.chmod(hostile, 0o777)  # AFTER mkdir: mkdir's mode argument is masked by the umask
    monkeypatch.setenv("CPO_AUTOTUNE_CACHE_DIR", str(hostile))
    cache_security.reset_cache()
    try:
        c = ResultCache("k", enabled=True)
        with pytest.warns(RuntimeWarning, match="refusing to use cache directory"):
            c.store(("m", 4096), {_A: 1.0})
        assert not list(hostile.glob("*.json")), "a refused root was written to"
        assert c.load(("m", 4096), [_A]) is None, "a refused root produced a hit"
    finally:
        cache_security.reset_cache()


def test_a_symlinked_entry_is_a_MISS_rather_than_a_read(cache, tmp_path):
    """The entry that decides a tile shape must be the file this user wrote, not a link to it."""
    cache.store(("m", 4096), {_A: 1.0})
    entry = list(tmp_path.glob("k.*.json"))[0]
    assert cache.load(("m", 4096), [_A]) is not None, "the positive control did not hit"
    elsewhere = tmp_path.parent / "elsewhere.json"
    elsewhere.write_text(entry.read_text())
    os.replace(str(entry), str(tmp_path.parent / "orig.json"))
    os.symlink(str(elsewhere), str(entry))
    assert cache.load(("m", 4096), [_A]) is None, "a symlinked cache entry was read"


@pytest.mark.parametrize("mode", [0o666, 0o620])
def test_a_WRITABLE_timing_entry_is_a_MISS_and_no_stale_timings_are_used(cache, tmp_path, mode):
    """Timings are not code, but an entry others can write still chooses this run's kernel config.

    The consequence is a silently mis-tuned job -- correct output, wrong tile shape, and nothing
    pointing at the cause because the sweep that would have caught it is the very thing being
    skipped. Refused rather than repaired, on the same reasoning as the artifact caches.
    """
    cache.store(("m", 4096), {_A: 1.0})
    entry = list(tmp_path.glob("k.*.json"))[0]
    assert cache.load(("m", 4096), [_A]) is not None, "the positive control did not hit"
    os.chmod(entry, mode)
    assert cache.load(("m", 4096), [_A]) is None, f"a {mode:04o} timing entry was read"
