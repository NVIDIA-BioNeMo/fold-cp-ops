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

# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.
"""Persistent .o cache for CuTe DSL compiled kernels.

Compiled kernels are exported as object files (.o) via export_to_c.
On subsequent runs the .o is loaded via tvm_ffi (~1ms) instead of
re-generating IR + re-JIT'ing (~100ms per kernel).

Controls:
  CPO_CACHE_ENABLED=0       — disable persistent .o cache (default: enabled)
  CPO_CACHE_DIR=path        — override default cache directory
"""

import fcntl
import functools
import hashlib
import os
import re
import stat
import sys
import tempfile
import time
from collections import namedtuple
from getpass import getuser
from pathlib import Path

import cutlass
import cutlass.cute as cute
import tvm_ffi

from fold_cp_ops._internal.cache_security import (
    CacheRoot,
    ensure_private_file,
    validate_cache_root,
)
from fold_cp_ops._internal.compile_time.template_params import (
    UnsupportedKeyComponent,
    canonical_key_bytes,
)

CACHE_ENABLED: bool = os.getenv("CPO_CACHE_ENABLED", "1") == "1"
CACHE_DIR: str | None = os.getenv("CPO_CACHE_DIR", None)
COMPILE_ONLY: bool = False

# Downstream projects can append directories here to include their sources
# in the cache fingerprint. Must be set before the first jit_cache call.
EXTRA_SOURCE_DIRS: list[Path] = []

EXPORT_FUNC_NAME = "func"
LOCK_TIMEOUT = 60
CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def _noop_kernel(*args, **kwargs):
    pass


def cache_root() -> CacheRoot:
    """The JIT cache root and the verdict on whether it may be used.

    Purpose
        One place that decides both WHERE the ``.o`` cache lives and WHETHER this process is
        allowed to touch it, so the two can never disagree.

    Functionality & semantics
        The location is unchanged: ``$CPO_CACHE_DIR`` when set, else
        ``<tempdir>/<user>/fold_cp_ops_cache``. The default sits under a world-writable ``/tmp``
        and its name is guessable, which is why it is validated rather than merely created --
        `cache_security.validate_cache_root` makes it ``0700``, refuses it if somebody else owns it
        or others can write it, and memoises the verdict so this costs one ``stat`` per process
        rather than one per compile.

    Returns:
        A `cache_security.CacheRoot`. When ``usable`` is False the caller must skip disk caching
        entirely; `jit_cache` does, and still compiles and still caches in memory.
    """
    if CACHE_DIR is not None:
        cache_dir = Path(CACHE_DIR)
    else:
        cache_dir = Path(tempfile.gettempdir()) / getuser() / "fold_cp_ops_cache"
    return validate_cache_root(cache_dir)


def get_cache_path() -> Path:
    """The JIT cache directory, created ``0700``.

    Kept as a `Path`-returning function because callers and tests outside this module use it as a
    location, not as a permission. Anything that must not write to an unsafe root asks
    :func:`cache_root` instead and checks ``usable`` -- a path alone cannot carry that answer, which
    is why the verdict is a separate call rather than an exception from this one.

    Returns:
        The directory. It exists whenever it could be created; when it could not, the path is still
        returned (naming it is useful in a message) and :func:`cache_root` reports it unusable.
    """
    return cache_root().path


def _hash_source_dir(h, root: Path) -> None:
    """Hash all Python sources under *root* into *h*."""
    for src in sorted(root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)


@functools.lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """Hash the package + extra source dirs plus runtime ABI stamps into a fingerprint."""
    h = hashlib.sha256()
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"tvm_ffi={tvm_ffi.__version__}".encode())
    # The PACKAGE root, not this module's directory: the fingerprint must cover every kernel
    # module (`kernels/`, `distributed/`), or a kernel edit would not bust the disk cache and a
    # stale compiled artifact would be silently reused.  This module lives one level down in
    # `_internal/`, so the root is `parent.parent`.
    _hash_source_dir(h, Path(__file__).resolve().parent.parent)
    for extra_dir in EXTRA_SOURCE_DIRS:
        _hash_source_dir(h, Path(extra_dir).resolve())
    return h.hexdigest()


def _key_to_hash(key: tuple) -> str:
    """SHA-256 over the canonical, type-tagged MessagePack encoding of *key*.

    Purpose
        Turn a call's key tuple into the ``<sha>.o`` filename, deterministically and across
        processes.

    Functionality & semantics
        ``canonical_key_bytes`` TYPE-TAGS each component, so ``True``, ``1`` and ``1.0`` -- which
        are all ``==`` in Python -- cannot map to the same cache entry. Its canonical encoding is
        also stable across interpreter versions.

        The key schema is versioned inside those bytes, preventing incompatible encodings from
        sharing an artifact name.

    Args:
        key: The call key, a tuple of :func:`is_key_component` values.

    Returns:
        64 hex characters.

    Raises:
        UnsupportedKeyComponent: If any component is outside that domain. `jit_cache` catches it
            and BYPASSES the disk cache -- an unformable key is a miss, never a failed call.
    """
    return hashlib.sha256(canonical_key_bytes(key)).hexdigest()


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


class FileLock:
    """Advisory file lock using fcntl.flock with timeout."""

    def __init__(self, lock_path: Path, exclusive: bool, timeout: float = 15):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self.timeout = timeout
        self._fd: int = -1

    def __enter__(self) -> "FileLock":
        flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        # O_NOFOLLOW, and 0600 at CREATION rather than a chmod after.
        #
        # The lock is the least interesting file in the cache and therefore the one that gets
        # overlooked: it carries no data, so it reads as harmless. It is not. It is created on the
        # READ path too, a symlinked lock redirects this open outside the validated root, and a
        # lock somebody else can write is a lock somebody else can hold -- which stalls every
        # compile in this process until the timeout, or lets them observe which keys are being
        # built. The mode argument is ignored when the file already exists, which is why the
        # existing file is inspected below rather than assumed.
        flags |= os.O_NOFOLLOW
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        try:
            self._fd = os.open(str(self.lock_path), flags, 0o600)
        except OSError as exc:
            # Fail-soft through the SAME `RuntimeError` path the timeout uses, which `jit_cache`
            # already catches and turns into "compile normally". A symlinked or unopenable lock
            # must cost a cache miss, never a failed call.
            raise RuntimeError(f"Refusing lock {self.lock_path}: {exc}") from exc
        st = os.fstat(self._fd)
        bad = None
        if not stat.S_ISREG(st.st_mode):
            bad = "not a regular file"
        elif st.st_uid != os.geteuid():
            bad = f"owned by uid {st.st_uid}, not by this process"
        elif stat.S_IMODE(st.st_mode) & 0o022:
            bad = f"mode {stat.S_IMODE(st.st_mode):04o} is writable by group or others"
        if bad is not None:
            os.close(self._fd)
            self._fd = -1
            raise RuntimeError(f"Refusing lock {self.lock_path}: {bad}")
        if stat.S_IMODE(st.st_mode) != 0o600:
            try:
                os.fchmod(self._fd, 0o600)
            except OSError:
                pass
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                return self
            except OSError:
                time.sleep(0.1)
        os.close(self._fd)
        self._fd = -1
        raise RuntimeError(f"Timed out waiting for lock: {self.lock_path}")

    def __exit__(self, *exc) -> None:
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


# ---------------------------------------------------------------------------
# Reloaded-artifact ABI restoration
# ---------------------------------------------------------------------------
#
# cute.compile returns a TVMFFIJitCompiledFunctionWithKwargs whenever the @cute.jit entry has ANY
# defaulted parameter (mB2/mB3/mSFA/mA2/mGate3/varlen_args/...).  That object carries a kwargs
# wrapper built from the entry's Python signature and FILLS the omitted trailing params with their
# defaults before the C call -- which is why `compiled_fn(mA, mB, mD, mC, epi, sched, varlen)` works
# on a fresh compile even though the kernel declares two more slots.
#
# export_to_c -> load_module does NOT carry that wrapper: it hands back a bare tvm_ffi.Function, and
# the export strips the args-spec metadata (`<name>_args_spec` is not in preserve_symbols), so the
# signature cannot be recovered from the artifact.  Meanwhile the generated `__tvm_ffi_func` hard-
# checks `num_args == <ALL declared params minus the EnvStream>`.  Net effect: every call site that
# relies on the default-filling works on a cache MISS and raises `TypeError: Expects N parameters`
# on a cache HIT -- i.e. only ever in a second process against a warm cache.
#
# Fix: give the reloaded artifact back the fresh callable's convention by padding the omitted
# trailing slots with None (their compile-time value -- they were traced as ConstNone).
#
# The target arity is recorded in a `<sha>.abi` sidecar written beside the `.o` at EXPORT time, so
# the LOAD path is a pure file read -- no probing, no dependency on DSL internals.  An entry with no
# sidecar (written before this change) is reported as a MISS and recompiled, which rewrites the
# sidecar; that costs exactly one cold compile per legacy entry and never yields a mis-ABI callable.
#
# The arity itself cannot be read off the export side: `c_header_arguments` bails on the EnvStream
# param (`.arguments == []`, `.error_msg` set) and `get_kwargs_wrapper_spec()` COUNTS the EnvStream
# param, over-reporting by a number neither object exposes.  So it is measured once, on the cold
# compile path, by asking the just-exported artifact.  If that ever stops working the sidecar is
# simply not written and loads recompile -- caught by the hit/miss assertions in
# tests/test_jit_cache_reload_abi.py, never silently wrong.
_ABI_SUFFIX = ".abi"
_ARITY_RE = re.compile(r"Expects (\d+) parameters")
# A probe arity no kernel has.  Doubly safe: even if some future entry matched it, every arg is None
# and every entry has at least one required Tensor param, so decoding fails before the body runs.
_ARITY_PROBE = (None,) * 64


def _measure_exported_arity(o_path: Path):
    """Positional arity the exported artifact demands, or None if it can't be determined.

    Cold-compile path only.  The `num_args` check is the first op emitted in `__tvm_ffi_func`, ahead
    of any parameter decode, so a deliberately-impossible call is side-effect free.
    """
    try:
        loaded = cute.runtime.load_module(str(o_path), enable_tvm_ffi=True)[EXPORT_FUNC_NAME]
        loaded(*_ARITY_PROBE)
    except Exception as e:
        m = _ARITY_RE.search(str(e))
        if m is not None:
            n = int(m.group(1))
            if 0 < n < len(_ARITY_PROBE):
                return n
    return None


def _write_abi_sidecar(abi_path: Path, o_path: Path) -> None:
    """Record the exported arity beside the .o (atomically, so a torn file is never read)."""
    arity = _measure_exported_arity(o_path)
    if arity is None:
        print(f"fold_cp_ops cache: could not determine ABI arity for {o_path.name}; will recompile")
        return
    tmp = abi_path.with_suffix(abi_path.suffix + ".tmp")
    tmp.write_text(str(arity))
    # Tighten BEFORE the rename: after it the sidecar is visible to every reader, so a chmod that
    # follows publication leaves a window in which the published file is world-readable.
    os.chmod(tmp, 0o600)
    os.replace(tmp, abi_path)


def _read_abi_sidecar(abi_path: Path):
    try:
        n = int(abi_path.read_text().strip())
    except (OSError, ValueError):
        return None
    return n if n > 0 else None


def _restore_call_abi(loaded, arity: int):
    """Wrap a reloaded artifact so omitted trailing Optional params are padded with None.

    CALL-ONLY: the returned object is a plain closure, NOT the compiled-function object.  It forwards
    `__call__` and nothing else -- no `.to()`, no `.jit_module`, no `.export_to_c`.  An attribute
    access would work on a cache MISS (fresh object) and break on a HIT -- a new asymmetry of exactly
    the family this wrapper exists to fix.

    No `@jit_cache`'d caller does this today, but attribute access on a compiled object IS a live
    pattern elsewhere: the nvshmem bitcode registration in `distributed/a2a.py:269`,
    `distributed/fused_trimul.py:281` and `distributed/gemm_bitcode_compile.py:155` walks
    `compiled.to(dev).jit_module.cuda_library`.  Those sit on their OWN compile caches, not on
    `jit_cache`, which is the only reason they are unaffected -- so moving any of them onto
    `jit_cache` (tempting, since bitcode compiles are slow) requires forwarding those attributes
    here first.  (`jit_cache` itself is safe: step 4 exports from `compiled_fn`, the fresh object,
    reached only on a miss.)
    """

    def call(*args, **kwargs):
        if kwargs:
            raise TypeError(
                "fold_cp_ops cache: the reloaded artifact is positional-only (the compile-time kwargs "
                f"wrapper is not carried by export_to_c); pass all {arity} args positionally"
            )
        n = len(args)
        return loaded(*args) if n == arity else loaded(*args, *(None,) * (arity - n))

    return call


# ---------------------------------------------------------------------------
# JIT cache decorator
# ---------------------------------------------------------------------------


def jit_cache(fn):
    """Decorator that caches compiled CuTe DSL kernels in-memory and on disk.

    The decorated function should return a compiled kernel (i.e. call cute.compile).
    The disk cache key is (fn.__qualname__, *args, **sorted_kwargs).
    """
    cache = {}
    hits = 0
    misses = 0

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        cache_key = args + tuple(sorted(kwargs.items())) if kwargs else args

        # 1. In-memory hit
        if cache_key in cache:
            hits += 1
            return _noop_kernel if COMPILE_ONLY else cache[cache_key]

        # 2. Disk hit
        disk_key = (fn.__qualname__,) + cache_key
        # The FINGERPRINT SUBDIRECTORY is validated too, not just its parent. Validating only the
        # parent would leave the directory entries are actually read from unchecked, and it is the
        # one this process creates -- so it is also the one a pre-created hostile directory would
        # be waiting at. An unusable root means `disk_ok` stays False and every disk step below is
        # skipped; the in-memory cache and the compile path are untouched, so the run is slower and
        # nothing else.
        disk_ok = False
        if CACHE_ENABLED:
            root = cache_root()
            if root.usable:
                try:
                    sha = _key_to_hash(disk_key)
                except UnsupportedKeyComponent:
                    # A key this process cannot NAME is a cache miss, not an error. Falling through
                    # with `disk_ok` False keeps the in-memory cache and the ordinary compile path
                    # exactly as they were; only the disk half is skipped. Raising here would turn a
                    # merely-unkeyable argument into a failed call, which is the opposite of what a
                    # cache is for -- and silently keying it on `repr` would be worse still, since
                    # two different values can share one.
                    sha = None
                if sha is not None:
                    fingerprint_root = validate_cache_root(
                        root.path / _compute_source_fingerprint()
                    )
                    disk_ok = fingerprint_root.usable
        if disk_ok:
            cache_path = fingerprint_root.path
            o_path = cache_path / f"{sha}.o"
            abi_path = cache_path / f"{sha}{_ABI_SUFFIX}"
            lock_path = cache_path / f"{sha}.lock"
            try:
                with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
                    # export_to_c drops the compile-time kwargs wrapper, so the reloaded artifact
                    # demands the FULL declared arity.  Restore the fresh callable's convention from
                    # the recorded arity; with no sidecar, MISS and recompile (which writes one)
                    # rather than hand out a callable whose ABI we cannot vouch for.
                    #
                    # `ensure_private_file` REPLACES the bare `o_path.exists()` and is doing two
                    # jobs: it refuses a symlink or a file this user does not own -- the entry
                    # about to be handed to `load_module` and EXECUTED -- and it holds both files
                    # at 0600. A refusal is an ordinary miss, so a cache written before this check
                    # existed is re-made rather than stranded.
                    readable = ensure_private_file(o_path) and ensure_private_file(abi_path)
                    arity = _read_abi_sidecar(abi_path) if readable else None
                    if arity is not None:
                        m = cute.runtime.load_module(str(o_path), enable_tvm_ffi=True)
                        loaded = _restore_call_abi(m[EXPORT_FUNC_NAME], arity)
                        cache[cache_key] = loaded
                        hits += 1
                        return _noop_kernel if COMPILE_ONLY else loaded
            except RuntimeError:
                pass

        # 3. Compile
        misses += 1
        compiled_fn = fn(*args, **kwargs)

        # 4. Store
        cache[cache_key] = compiled_fn
        if disk_ok:
            try:
                with FileLock(lock_path, exclusive=True, timeout=LOCK_TIMEOUT):
                    # ALWAYS EXPORT FRESH; never adopt whatever is sitting at `o_path`.
                    #
                    # The old code did `if not o_path.exists(): export(...)`, which makes an
                    # existing file the authority on its own contents -- and this process has just
                    # decided (via `ensure_private_file` on the read path) that it does not trust
                    # what is there. Re-exporting costs a compile we have already paid for and
                    # removes the case where a foreign or unreadable entry is left in place and
                    # then handed to the next reader.
                    tmp_o = o_path.with_name(f"{o_path.name}.{os.getpid()}.tmp")
                    compiled_fn.export_to_c(
                        object_file_path=str(tmp_o), function_name=EXPORT_FUNC_NAME
                    )
                    os.chmod(tmp_o, 0o600)
                    # Arity from the FILE WE JUST WROTE, not from `o_path`: measuring the old entry
                    # and publishing the new one would record an ABI for different bytes.
                    arity = _measure_exported_arity(tmp_o)
                    tmp_abi = abi_path.with_name(f"{abi_path.name}.{os.getpid()}.tmp")
                    if arity is None:
                        print(
                            f"fold_cp_ops cache: could not determine ABI arity for {o_path.name}; "
                            f"will recompile"
                        )
                        os.replace(str(tmp_o), str(tmp_o) + ".unpublished")
                    else:
                        tmp_abi.write_text(str(arity))
                        os.chmod(tmp_abi, 0o600)
                        # Sidecar FIRST, artifact second. A reader requires both, and publishing in
                        # this order means a visible `.o` always has its `.abi` -- the same rule
                        # `artifact_cache.write_artifact` follows for its meta.
                        os.replace(str(tmp_abi), str(abi_path))
                        os.replace(str(tmp_o), str(o_path))
            except Exception as e:
                print(f"fold_cp_ops cache: export failed for key {sha}: {e}")

        return _noop_kernel if COMPILE_ONLY else compiled_fn

    def cache_clear():
        nonlocal hits, misses
        cache.clear()
        hits = 0
        misses = 0

    def cache_info():
        return CacheInfo(hits=hits, misses=misses, maxsize=None, currsize=len(cache))

    wrapper.cache = cache
    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper
