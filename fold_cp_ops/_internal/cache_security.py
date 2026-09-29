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

"""Make this library's persistent caches private to the user who created them.

WHAT THIS IS FOR. Three caches here persist compiled code and measured timings across processes:
the JIT ``.o`` cache (`cache_utils`), the artifact cache (`artifact_cache`) and the autotune result
cache (`autotune.cache`). Every one of them writes under a path a *local* user can often predict --
``$TMPDIR/<user>/fold_cp_ops_cache`` most of all, since ``/tmp`` is world-writable and the username
is not a secret. A cache entry is loaded as EXECUTABLE CODE (``cute.runtime.load_module``), so a
writable cache root is a code-execution primitive for anybody on the box, and a pre-created
world-writable directory at the expected path is enough to obtain one.

This is a DIFFERENT boundary from the one `SECURITY.md` draws around a process group. Ranks in a
job are mutually trusted; a stranger with a shell on the same node never joins that group, so the
group's trust says nothing about them. That gap is what this module closes.

THE SHAPE OF THE DEFENCE, and why each piece is the way it is:

- **Lexical normalisation, never ``realpath``.** The final component is exactly what an attacker
  would replace with a symlink, so resolving it before the check would validate the TARGET and then
  use the LINK. `os.path.abspath` normalises ``..`` and relative segments without following
  anything.
- **``O_DIRECTORY | O_NOFOLLOW``, then ``fstat`` the descriptor.** Checking a path and then opening
  it is two resolutions of one name with a window in between. Opening first and asking the
  DESCRIPTOR closes it: everything downstream is a fact about the inode actually held.
- **Group/world-writable roots are REFUSED, not repaired.** Tightening such a root is not a fix --
  a hostile entry may already be inside it, and an attacker holding an open descriptor keeps write
  access across the ``chmod``. Only an owner-owned root that is already unwritable by others is
  tightened (to ``0700``), which is a real no-op for anyone but the owner.
- **A rejected root DISABLES disk caching; it never fails the run.** These caches are an
  optimisation. Raising would convert a permissions oddity into an outage, and the failure mode of
  a cache that is off is a slow run, which is recoverable, while the failure mode of a cache that is
  on and hostile is not.
- **Results are memoised, accepted AND rejected alike, and a rejection warns once.** ``jit_cache``
  consults its root on every call; re-``stat``-ing per compile is waste, and a warning per compile
  is a log nobody reads. Memoising only the accepted case would leave the rejected path paying both
  costs forever -- which is the path where they hurt.

The module is stdlib-only ON PURPOSE. It is imported by `cache_utils`, which is imported before
anything else in the package sets up; a dependency here would widen that edge for no gain.
"""

from __future__ import annotations

import os
import stat
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

__all__ = [
    "CacheRoot",
    "ensure_private_file",
    "is_within",
    "reset_cache",
    "validate_cache_root",
]

#: Bits that must be clear on a cache root. ``0o022`` is group-write plus other-write; a root
#: carrying either can be written by somebody other than its owner, which is the whole hazard.
_OTHER_WRITE = 0o022

#: The mode a validated root is held at, and the mode a cache file is held at.
_ROOT_MODE = 0o700
_FILE_MODE = 0o600

_LOCK = threading.Lock()
_CACHE: Dict[str, "CacheRoot"] = {}


@dataclass(frozen=True)
class CacheRoot:
    """The verdict on one cache root, and the path it refers to.

    Purpose
        Let a caller ask "may I use this directory?" and get back both the answer and the
        normalised path in one object, so the two can never be carried separately and get out of
        step.

    Functionality & semantics
        ``usable`` is the verdict. ``path`` is the LEXICALLY normalised path and is populated even
        when ``usable`` is False -- a caller that wants to name the offending directory in a
        message needs it, and a rejected root is still a location, not an absence. ``reason`` is a
        one-line human explanation, empty when ``usable``. Truthiness follows ``usable``, so
        ``if root:`` reads correctly at a call site.

        Frozen, because it is memoised and handed to many callers; a mutable verdict shared across
        the process is a verdict that can be edited by whoever received it last.

    Input requirements
        path: An absolute, lexically normalised `Path`. Not checked -- this class is constructed
            only by :func:`validate_cache_root`, which normalises. Constructing one by hand with a
            relative path yields an object whose ``path`` does not mean what its holder assumes.
        usable: The verdict.
        reason: Required non-empty when ``usable`` is False, so a rejection can always be explained;
            empty otherwise.
    """

    path: Path
    usable: bool
    reason: str = ""

    def __bool__(self) -> bool:
        """True when the root passed validation, so ``if root:`` is the natural spelling."""
        return self.usable


def _normalize(path: Union[str, os.PathLike]) -> Path:
    """Absolute, lexically normalised form of *path*, following no symlink.

    Purpose
        Produce the memoisation key and the path every later check is performed against.

    Functionality & semantics
        `os.path.abspath` joins the cwd and collapses ``.`` / ``..`` textually. It does NOT resolve
        symlinks, which is the entire reason it is used here rather than `Path.resolve` --
        resolving would check the link's target and then operate on the link, so a root replaced by
        a symlink after the check would pass it.

        A consequence worth naming: two spellings of one directory that differ through a symlink
        normalise differently and are validated (and memoised) separately. That is the safe
        direction -- an extra validation costs one ``open``, whereas coalescing them would let a
        checked name vouch for an unchecked one.

    Args:
        path: Any path-like. May be relative; it is joined to the current working directory, so a
            caller that later changes directory gets a DIFFERENT root from the same string.

    Returns:
        The normalised absolute `Path`.
    """
    return Path(os.path.abspath(os.fspath(path)))


def _reject(path: Path, reason: str, warn: bool) -> "CacheRoot":
    """Build a rejected verdict and, on first sight of this path, warn.

    Args:
        path: The normalised root being refused.
        reason: One line saying what is wrong with it.
        warn: Whether to emit the warning. False when the caller has already warned for this path
            (it is memoised), so a per-compile consultation does not produce a per-compile warning.

    Returns:
        An unusable :class:`CacheRoot` carrying *reason*.
    """
    if warn:
        warnings.warn(
            f"fold_cp_ops: refusing to use cache directory {path} ({reason}). Disk caching is "
            f"DISABLED for this process; compilation still works, it is just not reused. Point "
            f"CPO_CACHE_DIR (or the relevant CPO_* cache variable) at a directory you own, or fix "
            f"this one's ownership/permissions.",
            RuntimeWarning,
            stacklevel=3,
        )
    return CacheRoot(path=path, usable=False, reason=reason)


def validate_cache_root(path: Union[str, os.PathLike]) -> CacheRoot:
    """Create *path* ``0700`` if absent, verify it is private to this user, and memoise the verdict.

    Purpose
        The single gate every persistent cache in this package passes its root through, so the
        rules live in one place rather than being re-implemented (and diverging) in three.

    Functionality & semantics
        Normalises lexically (see :func:`_normalize`), then ``os.makedirs(mode=0o700,
        exist_ok=True)``. ``0o700`` survives any ``umask`` because it sets no group or other bit, so
        the process-wide ``umask`` is neither read nor written -- changing it would be a global
        side effect for a local need, and would race with any other thread creating a file.

        It then opens the result with ``O_DIRECTORY | O_NOFOLLOW`` and judges the DESCRIPTOR:

        ===============================  =========================================================
        condition                        verdict
        ===============================  =========================================================
        final component is a symlink     REJECT (``ELOOP`` from ``O_NOFOLLOW``)
        not a directory                  REJECT (``ENOTDIR`` from ``O_DIRECTORY``)
        ``st_uid != os.geteuid()``       REJECT -- somebody else owns it and can rewrite it
        group- or other-WRITABLE         REJECT -- unsafe to repair; see the module docstring
        owner-owned, others cannot       ACCEPT, and ``fchmod`` to ``0700`` if any g/o bit is set
        write, mode wider than 0700      (tightening the descriptor already held, not the path)
        exactly ``0700``, owner-owned    ACCEPT unchanged
        ===============================  =========================================================

        The verdict -- accepted or rejected -- is memoised on the normalised path string for the
        life of the process, and a rejection warns exactly once. Both are deliberate: `jit_cache`
        consults the root on every compile.

        A failure to create the directory at all (a read-only filesystem, a missing parent that
        cannot be made) is a REJECTION, not an exception. See the module docstring.

    Args:
        path: The intended cache root. Need not exist. May be relative, in which case it is
            resolved against the CURRENT working directory at the moment of the first call for that
            string -- so a caller that chdirs between calls is asking about two different roots and
            will get two different verdicts, correctly.

    Returns:
        A :class:`CacheRoot`. ``usable`` False means the caller MUST NOT read from or write to it;
        every caller in this package responds by skipping disk cache entirely and compiling
        normally. Never ``None``, so a caller cannot forget to check by writing ``if root is not
        None``.

    Raises:
        Nothing. Every failure is reported as an unusable verdict; a cache is an optimisation and
        must not be able to fail a run.
    """
    root = _normalize(path)
    key = str(root)
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    verdict = _validate_uncached(root)
    with _LOCK:
        # A concurrent caller may have finished first. Keep the winner rather than overwrite, so
        # every holder of this key sees one object and one warning was emitted, not two.
        verdict = _CACHE.setdefault(key, verdict)
    return verdict


def _check_ancestor(st: os.stat_result, path: Path) -> Optional[str]:
    """Why *st* is unacceptable as an ANCESTOR of a cache root, or ``None`` if it is fine.

    Purpose
        Validating only the final directory is not enough: anyone who can rename or replace a
        PARENT can substitute the whole subtree underneath it, so a ``0700`` leaf inside a
        world-writable parent is not private at all.

    Functionality & semantics
        An ancestor is accepted when it is a directory owned by this process's effective uid **or by
        root**, and no untrusted party can write to it. The root-owned case is what makes the
        default usable at all -- ``/``, ``/home`` and ``/tmp`` belong to root, not to the user.

        The sticky bit is the exception that matters. ``/tmp`` is mode ``1777``: world-writable, and
        safe anyway, because sticky means only an entry's OWNER may rename or remove it. Refusing
        every world-writable ancestor would reject the default cache location on every Linux box;
        accepting one that is world-writable and NOT sticky would accept a directory any user can
        swap out from under us. So the check is "not writable by others, UNLESS sticky".

    Args:
        st: ``os.fstat`` of the opened component -- the descriptor, never a re-resolved path.
        path: The component, for the message only.

    Returns:
        A one-line reason, or ``None`` when the ancestor is acceptable.
    """
    if not stat.S_ISDIR(st.st_mode):
        return f"{path} is not a directory"
    euid = os.geteuid()
    if st.st_uid not in (euid, 0):
        return f"{path} is owned by uid {st.st_uid}, neither this user ({euid}) nor root"
    mode = stat.S_IMODE(st.st_mode)
    if mode & _OTHER_WRITE and not (mode & stat.S_ISVTX):
        return f"{path} has mode {mode:04o}: writable by group or others and not sticky"
    return None


def _validate_uncached(root: Path) -> "CacheRoot":
    """The body of :func:`validate_cache_root`, without memoisation. Warns on rejection.

    Walks EVERY component from ``/`` downward, holding each parent's descriptor while opening or
    creating its child, so no component is ever resolved twice. That is what closes the window a
    path-at-a-time check leaves: between "stat the parent" and "open the child" an attacker with
    write access to the parent can swap the child, and the checked inode is not the used one.

    Args:
        root: An already-normalised absolute path.

    Returns:
        The verdict. See :func:`validate_cache_root` for the rules.
    """
    parts = root.parts
    if not parts or parts[0] != os.sep:
        return _reject(root, "not an absolute path", warn=True)

    fds = []
    try:
        try:
            fds.append(os.open(os.sep, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW))
        except OSError as exc:
            return _reject(root, f"cannot open '/': {exc}", warn=True)

        walked = Path(os.sep)
        for i, name in enumerate(parts[1:], start=1):
            walked = walked / name
            is_final = i == len(parts) - 1
            parent = fds[-1]
            try:
                # O_NOFOLLOW at EVERY level: a symlink anywhere in the chain redirects the leaf,
                # and following one here would validate a directory we are not going to use.
                fd = os.open(
                    name, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                )
            except FileNotFoundError:
                # Create it RELATIVE to the held parent descriptor, so the directory we make is
                # inside the directory we just validated -- not inside whatever that path names by
                # the time a second resolution happens.
                try:
                    os.mkdir(name, 0o700, dir_fd=parent)
                    fd = os.open(
                        name, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                    )
                except OSError as exc:
                    return _reject(root, f"cannot create {walked}: {exc}", warn=True)
            except OSError as exc:
                if os.path.islink(walked):
                    return _reject(root, f"{walked} is a symlink ({exc.strerror})", warn=True)
                return _reject(root, f"cannot open {walked} as a directory: {exc}", warn=True)
            fds.append(fd)

            st = os.fstat(fd)
            if not is_final:
                bad = _check_ancestor(st, walked)
                if bad is not None:
                    return _reject(root, bad, warn=True)
                continue

            # THE FINAL ROOT is held to a stricter bar than its ancestors: this is the directory
            # entries are read from and executed, so root ownership is not good enough and the
            # sticky exemption does not apply.
            euid = os.geteuid()
            if st.st_uid != euid:
                return _reject(
                    root, f"owned by uid {st.st_uid}, not by this process (uid {euid})", warn=True
                )
            mode = stat.S_IMODE(st.st_mode)
            if mode & _OTHER_WRITE:
                return _reject(root, f"mode {mode:04o} is writable by group or others", warn=True)
            if mode != _ROOT_MODE:
                # Safe to tighten: owner-owned and already unwritable by anyone else, so no hostile
                # entry can have been planted and no third party holds a write descriptor to lose.
                # `fchmod` needs a real descriptor -- O_PATH does not accept it -- so the directory
                # is reopened read-only THROUGH the validated parent rather than by path.
                try:
                    rw = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                except OSError as exc:
                    return _reject(root, f"cannot reopen to tighten mode: {exc}", warn=True)
                try:
                    if os.fstat(rw).st_ino != st.st_ino:
                        return _reject(root, "the directory changed during validation", warn=True)
                    os.fchmod(rw, _ROOT_MODE)
                except OSError as exc:
                    return _reject(
                        root, f"cannot tighten mode {mode:04o} to 0700: {exc}", warn=True
                    )
                finally:
                    os.close(rw)
    finally:
        for fd in fds:
            os.close(fd)

    return CacheRoot(path=root, usable=True)


def ensure_private_file(path: Union[str, os.PathLike]) -> bool:
    """Accept *path* only as an owner-owned regular non-symlink file, and hold it at ``0600``.

    Purpose
        The per-FILE half of the same check. A private root makes a planted entry hard; this makes
        an entry that somehow exists anyway -- an inherited file, a root validated after the fact,
        a directory shared through a bind mount -- refusable at the point it would be READ.

    Functionality & semantics
        Opens with ``O_NOFOLLOW`` and judges the descriptor, for the reason given in the module
        docstring. Accepts only a regular file owned by ``os.geteuid()``.

        **A file that was ALREADY group- or world-writable is refused outright, not repaired.** The
        distinction is the same one the root check draws, and it matters more here: if others could
        write the file, its CONTENT is already suspect, and a ``chmod`` afterwards changes only who
        may write it NEXT -- it says nothing about who wrote what is in it now, and an attacker
        holding an open descriptor keeps writing through the mode change. Tightening such a file
        would produce a private-looking entry with untrusted contents, which is strictly worse than
        a cache miss.

        A merely WIDE-BUT-READ-ONLY file (say ``0644``) is different: nobody else could have written
        it, so it is owner-authored, and it is tightened to ``0600`` rather than refused. That is
        what keeps an upgrade from stranding every entry written before this check existed.

    Args:
        path: The file to check. A missing file is a normal, quiet False -- callers use this as a
            read gate, and "not there" is the ordinary case, not an anomaly.

    Returns:
        True when the file exists, is a regular non-symlink file owned by this process's effective
        uid, and is now mode ``0600``. False otherwise, including when it does not exist. A False
        means "treat this as a cache miss"; it never means "raise".

    Raises:
        Nothing.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return False
        if st.st_uid != os.geteuid():
            return False
        mode = stat.S_IMODE(st.st_mode)
        if mode & _OTHER_WRITE:
            # Refuse, do NOT repair: see the docstring. Somebody else could have written this file,
            # so tightening it would only make untrusted content look private.
            return False
        if mode != _FILE_MODE:
            try:
                os.fchmod(fd, _FILE_MODE)
            except OSError:
                return False
        return True
    finally:
        os.close(fd)


def is_within(root: Union[str, os.PathLike], path: Union[str, os.PathLike]) -> bool:
    """Whether *path* lies under *root*, decided LEXICALLY.

    Purpose
        Stop a key or a filename that carries ``..`` (or an absolute path) from steering a read
        outside the root that was validated. Validating a root buys nothing if the entry read from
        it can name a file elsewhere.

    Functionality & semantics
        Both sides are normalised with `os.path.abspath` and compared by path parts. No symlink is
        resolved, for the same reason as everywhere else in this module: `Path.resolve` would judge
        a link's target while the caller then opens the link. A symlinked entry INSIDE the root is
        caught separately, by :func:`ensure_private_file`'s ``O_NOFOLLOW``, which is where that
        check belongs.

        ``is_within(root, root)`` is True: a root is within itself, and callers pass the root's own
        path when checking a directory-level operation.

    Args:
        root: The validated cache root.
        path: The candidate entry. Relative paths are resolved against the current working
            directory, which is almost never what a caller means -- pass an absolute path.

    Returns:
        True when *path* equals *root* or is nested under it.
    """
    r = _normalize(root).parts
    p = _normalize(path).parts
    return len(p) >= len(r) and p[: len(r)] == r


def reset_cache() -> None:
    """Forget every memoised verdict.

    Purpose
        Tests create a root, validate it, then change its ownership or mode and re-validate. Without
        this the second call returns the first call's answer and the test proves nothing.

    Functionality & semantics
        Clears the process-wide verdict table, so the next :func:`validate_cache_root` re-``stat``s
        and may warn again. Not for production use: a running process has no reason to re-decide,
        and clearing between compiles would restore exactly the per-compile ``stat`` and per-compile
        warning the memoisation exists to remove.

    Returns:
        None.
    """
    with _LOCK:
        _CACHE.clear()
