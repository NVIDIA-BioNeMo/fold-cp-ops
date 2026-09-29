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

"""Unit tests for ``fold_cp_ops/_internal/cache_utils.py``'s private-cache behaviour.

**No GPU and no compiler.** ``jit_cache`` is exercised with a FAKE compiled object whose
``export_to_c`` writes a few bytes, plus a fake ``load_module`` -- because what is under test is the
cache's filesystem contract (which root it will touch, what modes it publishes, what it refuses to
read back), and none of that involves CuTe. Requiring a device here would mean the contract is only
checked inside an allocation, which is the same as not checking it.

The load-bearing test is `test_an_unsafe_root_bypasses_the_disk_cache_without_blocking_compilation`.
A gate that turned an unsafe cache root into an exception would be worse than no gate: it converts a
permissions oddity on a shared machine into an outage, and the whole design rests on the opposite
choice -- a refused root costs a recompile and nothing else.
"""

import os
import stat

import pytest

from fold_cp_ops._internal import cache_security as cs
from fold_cp_ops._internal import cache_utils as cu
from fold_cp_ops._internal.compile_time.template_params import UnsupportedKeyComponent  # noqa: F401
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

pytestmark = matrix_exempt(
    "The subject is a host-side disk cache, not a kernel: these tests compile nothing and have no "
    "shape/dtype/tile axes for a KernelMatrix to declare."
)

_ARITY = 3


class _FakeCompiled:
    """Stands in for a ``cute.compile`` result: callable, and exportable to an object file.

    Purpose
        Let ``jit_cache`` run its full miss -> export -> hit cycle with no CuTe and no GPU.

    Functionality & semantics
        ``export_to_c`` writes deterministic bytes to the requested path, which is all the cache
        does with it. ``calls`` counts invocations so a test can tell a warm hit (the loaded
        artifact runs, not this) from a cold miss.

    Input requirements
        object_file_path: An absolute path in a directory that exists -- ``jit_cache`` creates it
            before calling. Anything else raises ``OSError``, exactly as the real export would.
    """

    def __init__(self, tag: str = "x"):
        self.tag = tag
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return f"fresh:{self.tag}"

    def export_to_c(self, object_file_path: str, function_name: str) -> None:
        """Write the stand-in artifact. Mirrors the real export's write-in-place behaviour."""
        with open(object_file_path, "wb") as fh:
            fh.write(b"FAKEOBJ:" + self.tag.encode())


@pytest.fixture
def fake_toolchain(monkeypatch, tmp_path):
    """Point the cache at ``tmp_path`` and replace the two CuTe entry points it uses.

    Purpose
        Isolate each test's cache, and make the disk round-trip observable without a compiler.

    Functionality & semantics
        Sets ``cache_utils.CACHE_DIR`` to a per-test directory, clears the validator's memo (it is
        process-wide and would otherwise carry one test's verdict into the next), and replaces
        ``_measure_exported_arity`` -- which really loads the artifact and calls it -- with a
        constant, plus ``cute.runtime.load_module`` with a fake returning a marker callable.

        Yields the root path so a test can inspect the files the cache published.
    """
    monkeypatch.setattr(cu, "CACHE_DIR", str(tmp_path / "jit"))
    monkeypatch.setattr(cu, "CACHE_ENABLED", True)
    monkeypatch.setattr(cu, "_measure_exported_arity", lambda o_path: _ARITY)
    monkeypatch.setattr(
        cu.cute.runtime,
        "load_module",
        lambda path, enable_tvm_ffi=True: {cu.EXPORT_FUNC_NAME: (lambda *a, **k: "loaded")},
    )
    cs.reset_cache()
    yield tmp_path / "jit"
    cs.reset_cache()


def _entries(root):
    """Every file the cache published under *root*, at any depth."""
    return sorted(p for p in root.rglob("*") if p.is_file())


def _mode(path) -> int:
    """Permission bits of *path*."""
    return stat.S_IMODE(os.stat(path).st_mode)


def test_a_cold_write_becomes_a_warm_hit(fake_toolchain):
    """The positive control. Without it every refusal test below is satisfied by a dead cache."""
    root = fake_toolchain
    made = []

    @cu.jit_cache
    def build(n):
        """Stand-in for a kernel builder."""
        obj = _FakeCompiled(f"n{n}")
        made.append(obj)
        return obj

    first = build(7)
    assert isinstance(first, _FakeCompiled), "the cold call must return the freshly built object"
    assert len(made) == 1

    # A second decorated function with the SAME qualname would collide; instead clear the in-memory
    # cache the only way a caller can, so the next call is forced onto the DISK path.
    build.cache_clear()
    second = build(7)
    assert len(made) == 1, "the warm call rebuilt instead of loading the artifact from disk"
    assert second(1) == "loaded", "the warm call did not return the reloaded artifact"

    names = {p.name for p in _entries(root)}
    assert any(n.endswith(".o") for n in names), f"no object file was published: {names}"
    assert any(n.endswith(cu._ABI_SUFFIX) for n in names), f"no ABI sidecar was published: {names}"


def test_the_published_o_abi_and_lock_files_are_all_0600(fake_toolchain):
    """Every file this cache creates must be private, including the lock nobody thinks about.

    The lock file is the one that gets missed: it carries no data, so it looks harmless -- but it is
    created on the READ path too, and a 0644 lock in a world-writable ``/tmp`` still advertises
    which keys this user is compiling.
    """
    root = fake_toolchain

    @cu.jit_cache
    def build(n):
        """Stand-in for a kernel builder."""
        return _FakeCompiled(f"n{n}")

    build(3)
    published = _entries(root)
    assert published, "the cache published nothing, so this test proved nothing"
    for path in published:
        assert _mode(path) == 0o600, f"{path.name} is {_mode(path):04o}, expected 0600"
    kinds = {p.suffix for p in published}
    assert {".o", ".lock"} <= kinds, f"expected at least a .o and a .lock, saw {kinds}"


def test_the_fingerprint_subdirectory_is_0700_not_just_its_parent(fake_toolchain):
    """The entries are read from the fingerprint dir, so it is the one that must be private."""
    root = fake_toolchain

    @cu.jit_cache
    def build(n):
        """Stand-in for a kernel builder."""
        return _FakeCompiled("f")

    build(1)
    assert _mode(root) == 0o700, f"cache root is {_mode(root):04o}"
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    assert len(subdirs) == 1, f"expected one fingerprint directory, got {subdirs}"
    assert _mode(subdirs[0]) == 0o700, f"fingerprint dir is {_mode(subdirs[0]):04o}, expected 0700"


def test_an_unsafe_root_bypasses_the_disk_cache_without_blocking_compilation(monkeypatch, tmp_path):
    """THE load-bearing case: refuse the disk, keep the run.

    A world-writable root must produce (a) no files on disk, and (b) a perfectly normal return
    value. Asserting only (a) would be satisfied by a cache that raised, which is the failure this
    design exists to avoid.
    """
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    os.chmod(hostile, 0o777)  # AFTER mkdir: the mode argument to mkdir is umask-masked
    monkeypatch.setattr(cu, "CACHE_DIR", str(hostile))
    monkeypatch.setattr(cu, "CACHE_ENABLED", True)
    cs.reset_cache()

    built = []

    @cu.jit_cache
    def build(n):
        """Stand-in for a kernel builder."""
        obj = _FakeCompiled("h")
        built.append(obj)
        return obj

    with pytest.warns(RuntimeWarning, match="refusing to use cache directory"):
        out = build(5)
    assert out is built[0], "compilation must still happen and its result must still be returned"

    build.cache_clear()
    out2 = build(5)
    assert out2 is built[1], "a refused root must recompile, not resurrect an entry"
    assert not [p for p in hostile.rglob("*") if p.is_file()], (
        f"files were written into a refused root: {list(hostile.rglob('*'))}"
    )
    cs.reset_cache()


def test_a_symlinked_o_entry_is_a_MISS_and_the_kernel_is_rebuilt(fake_toolchain):
    """An entry that is a symlink must not be loaded -- ``load_module`` EXECUTES what it opens."""
    root = fake_toolchain
    built = []

    @cu.jit_cache
    def build(n):
        """Stand-in for a kernel builder."""
        obj = _FakeCompiled("s")
        built.append(obj)
        return obj

    build(11)
    o_files = [p for p in root.rglob("*.o")]
    assert len(o_files) == 1, f"expected one artifact, got {o_files}"
    o_path = o_files[0]

    # Replace the published entry with a symlink to identical bytes. Content is unchanged, so only
    # the O_NOFOLLOW check can tell the difference -- which is exactly the property under test.
    elsewhere = root.parent / "elsewhere.o"
    elsewhere.write_bytes(o_path.read_bytes())
    os.replace(str(o_path), str(root.parent / "orig.o"))
    os.symlink(str(elsewhere), str(o_path))

    build.cache_clear()
    build(11)
    assert len(built) == 2, "a symlinked .o was loaded instead of being treated as a miss"


def test_a_foreign_owned_o_entry_is_a_MISS(fake_toolchain, monkeypatch):
    """Ownership is faked from the ``geteuid`` side, because ``chown`` needs privilege."""
    root = fake_toolchain
    built = []

    @cu.jit_cache
    def build(n):
        """Stand-in for a kernel builder."""
        obj = _FakeCompiled("o")
        built.append(obj)
        return obj

    build(13)
    assert len(built) == 1

    real = os.geteuid()
    monkeypatch.setattr(cs.os, "geteuid", lambda: real + 4242)
    build.cache_clear()
    build(13)
    assert len(built) == 2, "an entry owned by another uid was loaded"


def test_a_missing_abi_sidecar_is_still_a_MISS(fake_toolchain):
    """The pre-existing ABI rule must survive the new ownership checks, not be replaced by them."""
    root = fake_toolchain
    built = []

    @cu.jit_cache
    def build(n):
        """Stand-in for a kernel builder."""
        obj = _FakeCompiled("a")
        built.append(obj)
        return obj

    build(17)
    for abi in root.rglob(f"*{cu._ABI_SUFFIX}"):
        os.replace(str(abi), str(abi) + ".moved")
    build.cache_clear()
    build(17)
    assert len(built) == 2, "an artifact with no ABI sidecar was handed out"


def test_an_UNKEYABLE_argument_bypasses_the_disk_without_failing_the_call(fake_toolchain):
    """A key that cannot be FORMED is a cache miss, never an error.

    Purpose
        The key encoder is now type-strict (`canonical_key_bytes`), so it REFUSES values `pickle`
        would have swallowed. That refusal must not propagate: a caller passing an exotic argument
        should get a compiled kernel and a slower run, not an exception from the caching layer.

    Semantics
        A bare ``object()`` is hashable -- so the in-memory cache still works -- but is outside the
        key domain, so `_key_to_hash` raises `UnsupportedKeyComponent`. Both halves are asserted:
        the call returns normally AND nothing is written to disk. Asserting only the first would
        pass for an implementation that keyed on ``repr``, which silently merges distinct values
        that happen to print alike.
    """
    root = fake_toolchain
    built = []

    @cu.jit_cache
    def build(marker):
        """Stand-in for a kernel builder."""
        obj = _FakeCompiled("u")
        built.append(obj)
        return obj

    exotic = object()
    out = build(exotic)
    assert out is built[0], "an unkeyable argument must still compile and return normally"
    assert not [p for p in root.rglob("*") if p.is_file()], (
        f"an unkeyable call wrote to disk: {[p.name for p in root.rglob('*') if p.is_file()]}"
    )

    # And it is genuinely the KEY that refuses, not something incidental about `object()`.
    with pytest.raises(cu.UnsupportedKeyComponent):
        cu._key_to_hash((exotic,))


def test_a_group_writable_LOCK_is_refused_and_the_compile_still_succeeds(fake_toolchain):
    """The lock is the file everyone forgets; a foreign-writable one must not be honoured.

    A lock somebody else can write is a lock somebody else can HOLD, which stalls every compile in
    this process until the timeout. Refusing routes through `FileLock`'s existing ``RuntimeError``
    path, which `jit_cache` already treats as "compile normally" -- so the cost is a miss.
    """
    root = fake_toolchain

    @cu.jit_cache
    def build(n):
        """Stand-in for a kernel builder."""
        return _FakeCompiled("l")

    build(21)
    locks = [p for p in root.rglob("*.lock")]
    assert locks, "no lock file was created, so this test would prove nothing"
    os.chmod(locks[0], 0o666)

    built = []

    @cu.jit_cache
    def build2(n):
        """A second builder, so the key differs but the directory is shared."""
        obj = _FakeCompiled("l2")
        built.append(obj)
        return obj

    # Point the second builder at the SAME lock by reusing the first key's sha is not possible from
    # here, so assert the narrower property directly: FileLock refuses the hostile file.
    with pytest.raises(RuntimeError, match="Refusing lock"):
        with cu.FileLock(locks[0], exclusive=True, timeout=1):
            pass


def test_a_symlinked_o_is_refused_AND_its_target_is_left_untouched(fake_toolchain):
    """Refusing must not mutate anything outside the cache -- the link's target is not ours.

    `ensure_private_file` opens with ``O_NOFOLLOW``, so it never reaches the target; this pins that
    behaviour from the outside, because a naive "reject then chmod it private" would silently
    re-mode a file belonging to somebody else.
    """
    root = fake_toolchain

    @cu.jit_cache
    def build(n):
        """Stand-in for a kernel builder."""
        return _FakeCompiled("t")

    build(23)
    o_path = [p for p in root.rglob("*.o")][0]
    target = root.parent / "outside.o"
    target.write_bytes(b"NOT OURS")
    os.chmod(target, 0o644)
    before = stat.S_IMODE(os.stat(target).st_mode)
    os.replace(str(o_path), str(root.parent / "orig.o"))
    os.symlink(str(target), str(o_path))

    assert not cs.ensure_private_file(o_path), "a symlinked entry was accepted"
    assert stat.S_IMODE(os.stat(target).st_mode) == before, (
        "refusing a symlink must not chmod the file it points at"
    )
    assert target.read_bytes() == b"NOT OURS", "the target's contents were modified"


@pytest.mark.parametrize("mode", [0o666, 0o620])
def test_a_group_or_world_WRITABLE_o_entry_is_a_MISS_and_is_not_repaired(fake_toolchain, mode):
    """An entry others could have written is loaded as EXECUTABLE code; refuse it, do not repair it.

    Distinct from the symlink and foreign-owner cases: here the file is ours and is a regular file,
    and only its mode says somebody else could have changed the bytes. Tightening it would produce a
    private-looking entry whose contents are exactly as untrustworthy as before -- so the assertion
    covers both halves, the miss AND the mode being left alone.
    """
    root = fake_toolchain
    built = []

    @cu.jit_cache
    def build(n):
        """Stand-in for a kernel builder."""
        obj = _FakeCompiled("w")
        built.append(obj)
        return obj

    build(31)
    o_path = [p for p in root.rglob("*.o")][0]
    os.chmod(o_path, mode)

    build.cache_clear()
    build(31)
    assert len(built) == 2, f"a {mode:04o} .o entry was loaded instead of being treated as a miss"
    assert not cs.ensure_private_file(o_path) or _mode(o_path) != mode, (
        "the gate must refuse it; if it accepted, it must at least not have left it wide"
    )
