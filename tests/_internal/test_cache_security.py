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

"""Unit tests for ``fold_cp_ops/_internal/cache_security.py``.

**No GPU, and the negative controls are the point.** Every assertion here is about a REFUSAL --
a symlink, a file where a directory belongs, a directory somebody else owns, a directory the world
can write. A validator that accepted everything would satisfy every positive test in this file
(each of those uses a directory that is genuinely fine), so the refusals are the only ones that can
tell a working gate from an inert one.

Two things the tests have to work around, both of them properties of running as an ordinary user:

- **``mkdir(mode=...)`` is masked by the umask**, so a directory created with ``0o777`` is usually
  ``0o755`` and is NOT world-writable. Every permissive fixture therefore ``chmod``s AFTER creating.
  Getting this wrong does not fail the test -- it makes it pass for the wrong reason, which is how
  the first draft of this file "proved" that a world-writable root was accepted.
- **``chown`` needs privilege**, so the wrong-owner case fakes the OTHER side: ``os.geteuid`` is
  monkeypatched to a uid that is not the directory's. That is the same comparison the validator
  makes, approached from the end a test can actually reach.
"""

import os
import stat
import warnings
from pathlib import Path

import pytest

from fold_cp_ops._internal import cache_security as cs
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

pytestmark = matrix_exempt(
    "The subject is a filesystem permission gate, not a kernel: it compiles nothing and has no "
    "shape/dtype/tile axes for a KernelMatrix to declare."
)


@pytest.fixture(autouse=True)
def _clear_memo():
    """Drop the process-wide verdict table around every test.

    The validator memoises so that `jit_cache` pays one ``stat`` per process rather than one per
    compile. That is correct in production and ruinous here: a test that changes a directory's mode
    and re-validates would otherwise be handed the previous test's answer and assert nothing.
    Cleared BOTH before and after, so a test cannot inherit a verdict or leave one behind.
    """
    cs.reset_cache()
    yield
    cs.reset_cache()


@pytest.fixture
def permissive_umask():
    """Run the test under ``umask(0)``, restoring the process umask afterwards.

    ``0o700`` sets no group or other bit, so it survives any umask -- which is exactly the claim
    `validate_cache_root` rests on when it declines to touch the process-wide umask. Asserting that
    under the ambient ``0o022`` would prove nothing, because ``0o022`` cannot clear a bit that is
    already absent. Under ``umask(0)`` a wrong implementation (one that let the umask decide) would
    leave ``0o777``, so the assertion has something to fail on.
    """
    old = os.umask(0)
    try:
        yield
    finally:
        os.umask(old)


def _mode(path) -> int:
    """The permission bits of *path*, without the file-type bits."""
    return stat.S_IMODE(os.stat(path).st_mode)


def test_a_fresh_root_is_created_0700_even_under_a_permissive_umask(tmp_path, permissive_umask):
    """The created root must be 0700 because the MODE says so, not because the umask allowed it."""
    root = cs.validate_cache_root(tmp_path / "fresh")
    assert root.usable, f"a directory this process just created was refused: {root.reason}"
    assert _mode(root.path) == 0o700, (
        f"expected 0700 under umask(0), got {_mode(root.path):04o}; the umask is deciding the mode"
    )


def test_an_existing_owner_owned_root_is_TIGHTENED_rather_than_refused(tmp_path):
    """0755 owned by us is safe to repair: nobody else could have written into it.

    This is the case that separates "private" from "unusable". Refusing it would strand every cache
    written before this check existed; tightening it is a no-op for anyone but the owner.
    """
    d = tmp_path / "wide"
    d.mkdir()
    os.chmod(d, 0o755)
    root = cs.validate_cache_root(d)
    assert root.usable, f"an owner-owned, others-read-only root was refused: {root.reason}"
    assert _mode(d) == 0o700, f"expected the root to be tightened to 0700, got {_mode(d):04o}"


@pytest.mark.parametrize("mode", [0o770, 0o707, 0o777])
def test_a_group_or_world_writable_root_is_REFUSED_and_left_alone(tmp_path, mode):
    """Refused, not repaired -- and the mode must be untouched, which is the harder half.

    Tightening a root others can write is not a fix: a hostile entry may already be inside it, and
    an attacker holding an open descriptor keeps write access straight through the ``chmod``. So
    the test asserts the directory is left exactly as found; a validator that "helpfully" repaired
    it would pass a usable/unusable assertion alone.
    """
    d = tmp_path / f"perm{mode:04o}"
    d.mkdir()
    os.chmod(d, mode)  # AFTER mkdir: mkdir's mode argument is masked by the umask
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        root = cs.validate_cache_root(d)
    assert not root.usable, f"mode {mode:04o} is writable by others and was accepted"
    assert "writable by group or others" in root.reason, f"unexpected reason: {root.reason!r}"
    assert _mode(d) == mode, f"the refused root was modified: {mode:04o} -> {_mode(d):04o}"


def test_a_symlink_at_the_root_is_REFUSED_and_named_as_one(tmp_path):
    """The final component is what an attacker replaces, so it must never be followed.

    The reason string is asserted because the errno alone misleads: Linux reports ENOTDIR (not
    ELOOP) for ``O_NOFOLLOW`` combined with ``O_DIRECTORY``, so a message built from ``strerror``
    would call a symlink-to-a-directory "Not a directory".
    """
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    os.symlink(str(target), str(link))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        root = cs.validate_cache_root(link)
    assert not root.usable, "a symlinked cache root was accepted"
    assert "symlink" in root.reason, f"the refusal did not name the symlink: {root.reason!r}"
    assert os.path.islink(link), "the refused symlink must be left in place, not replaced"


def test_a_regular_file_where_the_root_belongs_is_REFUSED(tmp_path):
    """A file at the cache path is a refusal, not an exception the caller has to catch."""
    f = tmp_path / "notadir"
    f.write_text("x")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        root = cs.validate_cache_root(f)
    assert not root.usable, "a regular file was accepted as a cache root"
    assert root.path == f, "an unusable verdict must still name the path, for the message"


def test_a_root_owned_by_someone_else_is_REFUSED(tmp_path, monkeypatch):
    """Ownership is checked against ``os.geteuid()``, faked here because ``chown`` needs privilege.

    The comparison under test is ``st_uid != os.geteuid()``. A test cannot move ``st_uid`` without
    root, so it moves the other operand instead -- which exercises the same branch.
    """
    d = tmp_path / "theirs"
    d.mkdir()
    real = os.geteuid()
    monkeypatch.setattr(cs.os, "geteuid", lambda: real + 4242)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        root = cs.validate_cache_root(d)
    assert not root.usable, "a root owned by another uid was accepted"
    assert "owned by uid" in root.reason, f"unexpected reason: {root.reason!r}"


def test_an_uncreatable_root_is_a_REFUSAL_not_an_exception(tmp_path):
    """A cache is an optimisation; it must never be able to fail a run.

    A root under a directory this user cannot write is the ordinary way this happens in the field
    (a read-only filesystem, a removed parent). The call must return an unusable verdict.
    """
    parent = tmp_path / "locked"
    parent.mkdir()
    os.chmod(parent, 0o500)  # r-x: cannot create entries inside
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            root = cs.validate_cache_root(parent / "child")
        assert not root.usable, "a root that could not be created was reported usable"
        # The reason now names the COMPONENT that could not be made, because the walk knows
        # which one it was -- a chain rejection that cannot say where is hard to act on.
        assert "cannot create" in root.reason, f"unexpected reason: {root.reason!r}"
    finally:
        os.chmod(parent, 0o700)  # so tmp_path teardown can clean up


def test_the_verdict_is_memoised_for_an_ACCEPTED_root(tmp_path, monkeypatch):
    """A second consult must not re-stat -- `jit_cache` asks once per compile.

    Proved by making the underlying validation impossible to repeat: after the first call,
    ``os.open`` is replaced with one that raises. A non-memoising implementation would then report
    the root unusable.
    """
    d = tmp_path / "accepted"
    first = cs.validate_cache_root(d)
    assert first.usable

    def _explode(*a, **k):
        raise AssertionError("validate_cache_root re-validated a memoised root")

    monkeypatch.setattr(cs.os, "open", _explode)
    second = cs.validate_cache_root(d)
    assert second.usable and second.path == first.path


def test_a_REJECTED_root_is_memoised_too_and_warns_exactly_once(tmp_path):
    """Memoising only the accepted case leaves the rejected path paying every cost forever.

    The rejected path is the one where a per-compile ``stat`` and a per-compile warning hurt: a
    long sweep would emit thousands of identical RuntimeWarnings and bury everything else.
    """
    d = tmp_path / "hostile"
    d.mkdir()
    os.chmod(d, 0o777)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verdicts = [cs.validate_cache_root(d) for _ in range(5)]
    assert not any(v.usable for v in verdicts), "a world-writable root was accepted"
    hits = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(hits) == 1, f"expected exactly 1 warning across 5 consults, got {len(hits)}"
    assert "Disk caching is DISABLED" in str(hits[0].message), (
        f"the warning must say what it costs the user; got {str(hits[0].message)!r}"
    )


def test_reset_cache_makes_a_changed_root_visible_again(tmp_path):
    """Without this the whole file would be testing one verdict five ways."""
    d = tmp_path / "changes"
    d.mkdir()
    assert cs.validate_cache_root(d).usable
    os.chmod(d, 0o777)
    assert cs.validate_cache_root(d).usable, "the memo should still be in force"
    cs.reset_cache()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert not cs.validate_cache_root(d).usable, "reset_cache did not force a re-validation"


def test_two_lexical_spellings_of_one_root_are_validated_separately(tmp_path):
    """Normalisation is LEXICAL, so ``a/../a`` and ``a`` collapse but a symlinked alias does not.

    The safe direction: an extra validation costs one ``open``, whereas coalescing two names would
    let a checked one vouch for an unchecked one.
    """
    d = tmp_path / "a"
    d.mkdir()
    direct = cs.validate_cache_root(d)
    indirect = cs.validate_cache_root(tmp_path / "a" / ".." / "a")
    assert direct.path == indirect.path, "`..` should have collapsed lexically"
    assert indirect.usable


def test_ensure_private_file_accepts_an_owned_regular_file_and_tightens_it(tmp_path):
    """The per-file half: accept, and leave it 0600."""
    f = tmp_path / "entry.o"
    f.write_bytes(b"payload")
    os.chmod(f, 0o644)
    assert cs.ensure_private_file(f), "an owned regular file was refused"
    assert _mode(f) == 0o600, f"expected 0600, got {_mode(f):04o}"


def test_ensure_private_file_refuses_a_symlink_even_to_an_owned_file(tmp_path):
    """``O_NOFOLLOW``: the entry that gets EXECUTED must be the one that was checked."""
    real = tmp_path / "real.o"
    real.write_bytes(b"payload")
    link = tmp_path / "link.o"
    os.symlink(str(real), str(link))
    assert not cs.ensure_private_file(link), "a symlinked cache entry was accepted"


def test_ensure_private_file_refuses_a_directory_and_a_missing_path(tmp_path):
    """Both are quiet Falses: callers use this as a read gate, where "no" is the ordinary answer."""
    d = tmp_path / "adir"
    d.mkdir()
    assert not cs.ensure_private_file(d), "a directory was accepted as a cache entry"
    assert not cs.ensure_private_file(tmp_path / "absent.o"), "a missing file must be a quiet False"


def test_ensure_private_file_refuses_a_file_owned_by_someone_else(tmp_path, monkeypatch):
    """Same faked-uid technique as the root case, for the same reason."""
    f = tmp_path / "theirs.o"
    f.write_bytes(b"payload")
    real = os.geteuid()
    monkeypatch.setattr(cs.os, "geteuid", lambda: real + 4242)
    assert not cs.ensure_private_file(f), "a file owned by another uid was accepted"


def test_is_within_rejects_a_path_that_escapes_the_root(tmp_path):
    """A validated root buys nothing if an entry read from it can name a file elsewhere."""
    root = tmp_path / "root"
    root.mkdir()
    assert cs.is_within(root, root / "a.o")
    assert cs.is_within(root, root), "a root is within itself"
    assert not cs.is_within(root, tmp_path / "elsewhere" / "a.o")
    assert not cs.is_within(root, root / ".." / "escaped.o"), "`..` must not escape unnoticed"


def test_a_symlinked_ANCESTOR_is_refused_even_when_the_leaf_is_fine(tmp_path):
    """Validating only the leaf is the hole this walk exists to close.

    The leaf here is a perfectly good ``0700`` directory the user owns. What is wrong is one level
    up: an ancestor is a symlink, so whoever controls that link controls which subtree the "checked"
    leaf actually lives in. A leaf-only check passes this and uses the attacker's directory.
    """
    real = tmp_path / "real"
    (real / "leaf").mkdir(parents=True)
    link = tmp_path / "via_link"
    os.symlink(str(real), str(link))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        root = cs.validate_cache_root(link / "leaf")
    assert not root.usable, "a root reached through a symlinked ancestor was accepted"
    assert "symlink" in root.reason, root.reason


def test_a_WORLD_WRITABLE_non_sticky_ancestor_is_refused(tmp_path):
    """Anyone who can write the parent can replace the child, whatever the child's own mode says."""
    parent = tmp_path / "loose"
    leaf = parent / "cache"
    leaf.mkdir(parents=True)
    os.chmod(leaf, 0o700)
    os.chmod(parent, 0o777)  # world-writable and NOT sticky
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            root = cs.validate_cache_root(leaf)
        assert not root.usable, "a world-writable non-sticky ancestor was accepted"
        assert "sticky" in root.reason, root.reason
    finally:
        os.chmod(parent, 0o700)


def test_a_STICKY_world_writable_ancestor_is_ACCEPTED_because_that_is_slash_tmp(tmp_path):
    """Without this exemption the default cache root is unusable on every Linux box.

    ``/tmp`` is mode ``1777`` -- world-writable by design and safe anyway, because the sticky bit
    means only an entry's owner may rename or remove it. The test builds the same shape rather than
    touching the real ``/tmp``, so it asserts the RULE rather than the machine.
    """
    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(shared, 0o1777)
    try:
        root = cs.validate_cache_root(shared / "mine")
        assert root.usable, f"a sticky world-writable ancestor was refused: {root.reason}"
        assert _mode(root.path) == 0o700
    finally:
        os.chmod(shared, 0o700)


def test_the_real_default_cache_location_is_either_usable_or_refused_FOR_A_REAL_REASON(tmp_path):
    """End-to-end against the actual filesystem -- but the verdict is environmental, not a constant.

    Purpose
        Catch a rule that is right in a fixture and wrong on a real machine. A walk that refused
        every real ``/tmp`` would disable disk caching everywhere; one that accepted an unsafe chain
        would defeat the point.

    Why this does NOT assert ``usable``
        An earlier version did, and it FAILED on a real cluster -- correctly. There, ``/tmp`` is the
        expected root-owned ``1777`` but ``/tmp/<user>`` had been created ``0777`` WITHOUT the
        sticky bit, so any user on the node could rename or replace the cache directory inside it.
        Refusing that is the behaviour this module exists for, so the test was asserting a property
        of the machine and calling it a property of the code.

        What IS invariant is that the answer is reasoned: usable, or refused with a message naming
        an ancestor and the actual defect. A crash, an empty reason, or a refusal that cannot say
        which component was wrong would all be bugs here -- and are what this now checks.

    Args:
        tmp_path: Unused; kept so the fixture-scoped umask/memo handling stays uniform.
    """
    import getpass
    import tempfile

    target = Path(tempfile.gettempdir()) / getpass.getuser() / "fold_cp_ops_cache_probe"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        root = cs.validate_cache_root(target)

    if root.usable:
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)], (
            "an ACCEPTED root must not warn; a warning per process is one everyone learns to ignore"
        )
        assert _mode(root.path) == 0o700, f"an accepted root is {_mode(root.path):04o}, not 0700"
        return

    # Refused: the message must name a component and a recognised defect, so an operator can act.
    assert root.reason, "a refusal with no reason is unactionable"
    assert any(
        phrase in root.reason
        for phrase in ("symlink", "sticky", "owned by uid", "not a directory", "cannot create")
    ), f"the refusal reason is not one this module knows how to produce: {root.reason!r}"
    assert str(root.path) in str(target) or root.path == target, "the verdict names another path"
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)], (
        "a REFUSED root must warn -- disk caching just turned off and nothing else says so"
    )


def test_a_deeply_NESTED_root_is_created_and_every_new_level_is_0700(tmp_path, permissive_umask):
    """Creation happens relative to the held parent descriptor, level by level."""
    root = cs.validate_cache_root(tmp_path / "a" / "b" / "c" / "d")
    assert root.usable, root.reason
    for p in (tmp_path / "a", tmp_path / "a" / "b", tmp_path / "a" / "b" / "c", root.path):
        assert _mode(p) == 0o700, f"{p} is {_mode(p):04o}, expected 0700 under umask(0)"


@pytest.mark.parametrize("mode", [0o620, 0o602, 0o666])
def test_a_pre_existing_group_or_world_WRITABLE_file_is_refused_not_repaired(tmp_path, mode):
    """Its CONTENT is already suspect, so tightening it only makes untrusted bytes look private.

    This is the case where "helpfully fix it" is the wrong instinct: the mode change says who may
    write NEXT and nothing about who wrote what is there -- and an attacker holding an open
    descriptor writes straight through the ``chmod``.
    """
    f = tmp_path / "entry.o"
    f.write_bytes(b"payload")
    os.chmod(f, mode)
    assert not cs.ensure_private_file(f), f"a {mode:04o} file was accepted"
    assert _mode(f) == mode, "a REFUSED file must be left exactly as found, not repaired"


def test_a_wide_but_READ_ONLY_file_is_tightened_rather_than_refused(tmp_path):
    """Nobody else could have written it, so it is owner-authored -- refusing would strand caches.

    The counterpart to the test above, and the pair is the point: the two differ only in whether
    somebody ELSE could have written the file, which is exactly the property that decides whether
    its contents can be trusted.
    """
    f = tmp_path / "entry.o"
    f.write_bytes(b"payload")
    os.chmod(f, 0o644)
    assert cs.ensure_private_file(f), "an owner-authored 0644 file was refused"
    assert _mode(f) == 0o600


def test_a_verdict_is_memoised_only_AFTER_the_whole_chain_passed(tmp_path):
    """A rejection partway down the chain must not leave a usable verdict cached.

    The walk validates ancestors before the leaf, so an early rejection happens with the leaf
    unexamined. Caching anything usable at that point would hand the next caller an accept for a
    root whose chain was never finished.
    """
    parent = tmp_path / "loose"
    leaf = parent / "cache"
    leaf.mkdir(parents=True)
    os.chmod(parent, 0o777)
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            first = cs.validate_cache_root(leaf)
            second = cs.validate_cache_root(leaf)
        assert not first.usable and not second.usable, "a partially-walked chain became usable"
    finally:
        os.chmod(parent, 0o700)


@pytest.mark.parametrize("mode", [0o711, 0o701, 0o500])
def test_an_EXECUTE_ONLY_ancestor_is_accepted(tmp_path, mode):
    """Traversable-but-not-listable ancestors are normal and must not be refused.

    ``0711`` is the standard mode for a home directory on a shared box: others may traverse INTO it
    to reach a path they already know, but cannot list it and cannot write it. The rule under test
    is "not writable by others", not "not accessible to others" -- confusing the two would reject
    the ordinary layout of most multi-user systems and silently disable disk caching there.
    """
    parent = tmp_path / "traversable"
    leaf = parent / "cache"
    leaf.mkdir(parents=True)
    os.chmod(parent, mode)
    try:
        root = cs.validate_cache_root(leaf)
        assert root.usable, f"an execute-only ({mode:04o}) ancestor was refused: {root.reason}"
    finally:
        os.chmod(parent, 0o700)
