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


"""Tests for ``fold_cp_ops._internal.arch`` -- the SM90 capability boundary.

Every test here runs **without SM90 silicon**: the point of ``CPO_ARCH`` is that the refusal can be
exercised on any machine, including the one being refused.
"""

import pytest
import torch

from fold_cp_ops._internal import arch
from fold_cp_ops._internal.arch import (
    SUPPORTED_ARCH_MAJOR,
    UnsupportedArchError,
    _parse_arch_str,
    check_arch_supported,
    get_device_capacity,
    require_sm90,
)


@pytest.fixture(autouse=True)
def _clear_capacity_cache():
    """Clear the ``lru_cache`` around ``get_device_capacity`` on both sides of every test.

    Without this a test that sets ``CPO_ARCH`` would either read a value cached before it ran, or
    leave its own value cached for the next test -- and the second failure looks like a bug in
    whichever test happens to run after.
    """
    get_device_capacity.cache_clear()
    yield
    get_device_capacity.cache_clear()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("90", (9, 0)),
        ("sm90", (9, 0)),
        ("sm_90", (9, 0)),
        ("sm_90a", (9, 0)),
        ("SM_90A", (9, 0)),
        ("  sm_90  ", (9, 0)),
        ("100", (10, 0)),
        ("sm_100a", (10, 0)),
        ("sm_120f", (12, 0)),
        ("89", (8, 9)),
    ],
)
def test_parse_arch_str_accepts_every_spelling(text, expected):
    """All the ways CUDA tooling writes an architecture parse to the same pair.

    ``sm_100a -> (10, 0)`` is the case worth pinning: the split is "all but the last digit" / "last
    digit", so a naive two-character parse would read it as ``(1, 0)`` and silently pass the SM90
    gate on a Blackwell device.
    """
    assert _parse_arch_str(text) == expected


@pytest.mark.parametrize("text", ["", "sm", "gfx90a", "9", "sm_", "ninety", "sm_90b"])
def test_parse_arch_str_rejects_junk_by_name(text):
    """An unparseable value raises and names ``CPO_ARCH``, so the source of the string is obvious."""
    with pytest.raises(ValueError, match="CPO_ARCH"):
        _parse_arch_str(text)


def test_check_arch_supported_accepts_sm90_and_only_sm90():
    """Only major 9 passes; the minor is not consulted for the decision, only for the message."""
    check_arch_supported((SUPPORTED_ARCH_MAJOR, 0))
    check_arch_supported((SUPPORTED_ARCH_MAJOR, 7))
    for cap in [(8, 0), (10, 0), (12, 0), (7, 5)]:
        with pytest.raises(UnsupportedArchError, match=r"requires SM90 \(H100/H200\)"):
            check_arch_supported(cap)


def test_refusal_message_names_the_device_it_refused():
    """The message carries the capability it saw, so a log line identifies the machine."""
    with pytest.raises(UnsupportedArchError, match=r"got sm_100"):
        check_arch_supported((10, 0))


def test_unsupported_arch_error_is_not_an_assertion_error():
    """The boundary must survive ``python -O``, which strips ``assert`` but not ``raise``.

    Also checked: it is catchable as a distinct type, so a caller can fall back to a torch reference
    without matching on the message text.
    """
    assert issubclass(UnsupportedArchError, RuntimeError)
    assert not issubclass(UnsupportedArchError, AssertionError)


def test_cpo_arch_overrides_the_device_query(monkeypatch):
    """``CPO_ARCH`` wins over the real device, which is what makes the boundary testable anywhere."""
    monkeypatch.setenv("CPO_ARCH", "sm_100a")
    assert get_device_capacity() == (10, 0)
    with pytest.raises(UnsupportedArchError):
        require_sm90()


def test_require_sm90_returns_the_capability_for_reuse(monkeypatch):
    """On success it hands back the pair, so a caller can key a compile cache without re-querying."""
    monkeypatch.setenv("CPO_ARCH", "sm_90a")
    assert require_sm90() == (9, 0)


def test_require_sm90_gates_on_cpu_tensors_too(monkeypatch):
    """The gate runs before anything is traced, so it refuses without touching a GPU at all."""
    monkeypatch.setenv("CPO_ARCH", "sm_120a")
    cpu = torch.empty(0).device
    with pytest.raises(UnsupportedArchError, match="sm_120"):
        require_sm90(cpu)


def test_supported_arch_major_is_the_single_source_of_the_boundary():
    """One constant decides; raising it would not make an SM100 kernel exist, only move the failure.

    Pinned because the message and the comparison must stay in agreement -- a check against a
    literal 9 with the constant left at 9 reads fine and drifts silently.
    """
    assert SUPPORTED_ARCH_MAJOR == 9
    assert arch.SUPPORTED_ARCH_MAJOR is SUPPORTED_ARCH_MAJOR
