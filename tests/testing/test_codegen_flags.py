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

"""Tests for ``fold_cp_ops/testing/codegen_flags.py``.

No GPU: every behaviour that matters here is Python-level state management -- whether the memos are
cleared, whether the attribute is restored, and whether the cache-enabled case is refused. The
compilation hazard the module exists to prevent was established by measurement elsewhere (toggling
the flag with the cache ON produced no new ``.o``); these pin the guard, not the hazard.
"""

from __future__ import annotations

import pytest

from fold_cp_ops._internal import cache_utils
from fold_cp_ops.testing.codegen_flags import (
    clear_jit_caches,
    codegen_flag,
    iter_jit_caches,
)
from fold_cp_ops.testing.kernel_matrix import matrix_exempt


@pytest.fixture
def cache_off(monkeypatch):
    """Force ``cache_utils.CACHE_ENABLED`` False for a test.

    The module global is read at import time from ``CPO_CACHE_ENABLED``, so a test cannot influence
    it through the environment after the fact; patching the global is the only way to exercise both
    branches in one session.

    Yields:
        None. The original value is restored by ``monkeypatch``.
    """
    monkeypatch.setattr(cache_utils, "CACHE_ENABLED", False)
    yield


class _Base:
    """Stand-in for a kernel functor base carrying a codegen-affecting attribute."""

    FLAG = False


class _Derived(_Base):
    """Subclass that does NOT override ``FLAG``, so the inherited-attribute path is exercised."""


@matrix_exempt("guards a cache/state invariant; there is no kernel, shape or dtype to sweep")
def test_the_flag_is_set_inside_and_restored_outside(cache_off):
    """The temporary value is visible in the block and gone after it."""
    assert _Base.FLAG is False
    with codegen_flag(_Base, "FLAG", True):
        assert _Base.FLAG is True
    assert _Base.FLAG is False
    assert "FLAG" in vars(_Base)


@matrix_exempt("guards a cache/state invariant; there is no kernel, shape or dtype to sweep")
def test_an_inherited_attribute_is_deleted_rather_than_shadowed(cache_off):
    """Restoring an INHERITED attribute must remove the subclass binding, not rewrite it.

    Rewriting would leave ``_Derived`` with its own copy that merely equals ``_Base.FLAG`` today and
    silently stops tracking the base afterwards -- a divergence that no assertion on the value can
    see, which is why this asserts on ``vars()`` instead.
    """
    assert "FLAG" not in vars(_Derived)
    with codegen_flag(_Derived, "FLAG", True):
        assert _Derived.FLAG is True
        assert "FLAG" in vars(_Derived)
    assert "FLAG" not in vars(_Derived), "subclass kept a shadowing copy of an inherited attribute"
    assert _Derived.FLAG is False
    _Base.FLAG = "tracked"
    try:
        assert _Derived.FLAG == "tracked", "subclass stopped tracking its base"
    finally:
        _Base.FLAG = False


@matrix_exempt("guards a cache/state invariant; there is no kernel, shape or dtype to sweep")
def test_the_attribute_is_restored_even_when_the_body_raises(cache_off):
    """A failing test must not leak a codegen flag into every test that runs after it."""
    with pytest.raises(ValueError, match="boom"):
        with codegen_flag(_Base, "FLAG", True):
            raise ValueError("boom")
    assert _Base.FLAG is False


@matrix_exempt("guards a cache/state invariant; there is no kernel, shape or dtype to sweep")
def test_the_disk_cache_being_ON_is_refused_not_worked_around(monkeypatch):
    """With the cache enabled the helper raises, because the toggle would measure nothing.

    It refuses rather than disabling the cache itself: ``CACHE_ENABLED`` is read once at import into
    a module global, so flipping it here would not reach an already-imported reader, and a helper
    that looked like it fixed the problem would be worse than one that stops.
    """
    monkeypatch.setattr(cache_utils, "CACHE_ENABLED", True)
    with pytest.raises(RuntimeError, match="CPO_CACHE_ENABLED=0"):
        with codegen_flag(_Base, "FLAG", True):
            pass  # pragma: no cover - the context manager raises on entry
    assert _Base.FLAG is False


@matrix_exempt("guards a cache/state invariant; there is no kernel, shape or dtype to sweep")
def test_a_misspelled_attribute_is_refused(cache_off):
    """Creating an attribute nothing reads would present as 'the flag had no effect'."""
    with pytest.raises(AttributeError, match="no attribute"):
        with codegen_flag(_Base, "FLAGG", True):
            pass  # pragma: no cover - the context manager raises on entry


@matrix_exempt("guards a cache/state invariant; there is no kernel, shape or dtype to sweep")
def test_the_memos_are_cleared_on_entry_and_exit(cache_off):
    """Both clears are required: entry so the block compiles fresh, exit so it does not leak.

    Uses a fake wrapper carrying the three attributes ``jit_cache`` attaches, registered into an
    importable ``fold_cp_ops`` module, because the real compile entries need a GPU to populate.
    """
    import fold_cp_ops.testing.codegen_flags as mod

    calls = []

    def fake_entry():  # pragma: no cover - never invoked, only inspected
        pass

    fake_entry.cache = {}
    fake_entry.cache_info = lambda: None
    fake_entry.cache_clear = lambda: calls.append(1)

    mod._fake_jit_entry = fake_entry
    try:
        assert any(w is fake_entry for w in iter_jit_caches()), "discovery missed a marked wrapper"
        with codegen_flag(_Base, "FLAG", True):
            assert len(calls) == 1, "memos not cleared on entry"
        assert len(calls) == 2, "memos not cleared on exit"
    finally:
        del mod._fake_jit_entry


@matrix_exempt("guards a cache/state invariant; there is no kernel, shape or dtype to sweep")
def test_discovery_yields_each_wrapper_once_and_counts_what_it_cleared():
    """A wrapper re-exported from several modules must be yielded once, or counts are meaningless."""
    import fold_cp_ops.testing.codegen_flags as mod
    import fold_cp_ops.testing.kernel_matrix as other

    def fake_entry():  # pragma: no cover - never invoked, only inspected
        pass

    fake_entry.cache = {}
    fake_entry.cache_info = lambda: None
    fake_entry.cache_clear = lambda: None

    mod._fake_dup = fake_entry
    other._fake_dup = fake_entry
    try:
        assert sum(1 for w in iter_jit_caches() if w is fake_entry) == 1
        assert clear_jit_caches() == len(list(iter_jit_caches()))
    finally:
        del mod._fake_dup
        del other._fake_dup
