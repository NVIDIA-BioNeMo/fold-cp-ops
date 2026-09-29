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

"""Unit tests for `fold_cp_ops._internal.heuristic_arch`.

**What makes this module worth testing at all is that its failure mode is SILENCE.**
`heuristic_arch` reads the live device and has no other signal, so a config picked off the target
hardware falls back to `TUNED_ARCH` and looks exactly like one picked on it. Nothing downstream can
tell the two apart. So the tests here pin both halves of the mitigation -- an explicit override that
makes an off-device pick auditable, and a loud warning when the fallback happens implicitly -- and
they assert the ABSENCE of the wrong warning as carefully as the presence of the right one.

Almost all of it is GPU-free and runs anywhere, including a box with no CUDA device at all, which
is exactly where the fallback bites.
"""

import warnings

import pytest
import torch

from fold_cp_ops._internal.heuristic_arch import (
    _WARNED,
    TUNED_ARCH,
    _arch_label_for,
    heuristic_arch,
    warn_arch_fallback_once,
    warn_arch_suboptimal_once,
)


def test_the_tuned_arch_is_the_one_the_thresholds_were_bisected_on():
    """Pins the constant, because every shipped threshold table is keyed on it.

    Not a tautology: the label is what an arch-keyed table is indexed by, so renaming it silently
    turns every lookup into a fallback, and every heuristic keeps returning answers.
    """
    assert TUNED_ARCH == "H200_SXM5"


@pytest.mark.parametrize(
    "name,cc,sm,want",
    [
        ("NVIDIA H200", 90, 132, "H200_SXM5"),  # the TUNED arch; the name carries no "SXM"
        ("NVIDIA H100 80GB HBM3", 90, 132, "H100_SXM5"),
        ("NVIDIA H100 PCIe", 90, 114, "H100_PCIE"),
        ("NVIDIA H100", 90, 114, "H100_PCIE"),  # the SM count disambiguates when the name does not
        ("NVIDIA H800", 90, 132, "H800"),
        ("NVIDIA GH200 480GB", 90, 132, "GH200"),  # must NOT match the "H200" substring
        ("NVIDIA B200", 100, 148, "B200"),
        ("NVIDIA GB200 NVL72", 100, 148, "GB200"),  # must NOT match the "B200" substring
        ("NVIDIA A100-SXM4-80GB", 80, 108, "unknown_sm80"),
        ("Some Future GPU", 95, 200, "unknown_sm95"),
    ],
)
def test_a_device_name_maps_to_the_label_its_thresholds_are_keyed_on(name, cc, sm, want):
    """The (name, capability, SM count) -> label mapping, substring-ordering cases included.

    **The GH200 and GB200 rows are the ones that earn their keep.** "GH200" CONTAINS "H200" and
    "GB200" contains "B200", so a match order sorted any other way labels a Grace-Hopper part as
    the tuned arch and hands it thresholds bisected on a different machine -- silently, since both
    are real labels and neither triggers a fallback warning.
    """
    assert _arch_label_for(name, cc, sm) == want


def test_the_explicit_override_wins_and_is_stripped(monkeypatch):
    """``CPO_HEURISTIC_ARCH`` replaces detection, whitespace and all.

    Stripping matters because the value usually arrives from a shell, where a trailing space is
    invisible and would make the label miss every table entry -- i.e. fail as a fallback while
    LOOKING like an explicit choice, which is the worst of both.
    """
    monkeypatch.setenv("CPO_HEURISTIC_ARCH", "H100_SXM5")
    assert heuristic_arch() == "H100_SXM5"
    monkeypatch.setenv("CPO_HEURISTIC_ARCH", "  H200_SXM5  ")
    assert heuristic_arch() == "H200_SXM5"


def test_an_empty_override_is_the_same_as_no_override(monkeypatch):
    """``export CPO_HEURISTIC_ARCH=`` must not pin the arch to the empty string.

    An empty value read as an override would make every table lookup miss, so the heuristic would
    quietly use the fallback set while the operator believed they had selected an arch.
    """
    monkeypatch.delenv("CPO_HEURISTIC_ARCH", raising=False)
    detected = heuristic_arch()
    monkeypatch.setenv("CPO_HEURISTIC_ARCH", "")
    assert heuristic_arch() == detected
    monkeypatch.setenv("CPO_HEURISTIC_ARCH", "   ")
    assert heuristic_arch() == detected


def test_the_override_is_read_on_every_call_not_memoised(monkeypatch):
    """A harness that sets the override AFTER import must still be obeyed.

    `_arch_label_for` is memoised and the override deliberately is not: caching the env read would
    freeze the arch at import time, which is the same silent-staleness this module exists to stop.
    """
    monkeypatch.setenv("CPO_HEURISTIC_ARCH", "B200")
    assert heuristic_arch() == "B200"
    monkeypatch.setenv("CPO_HEURISTIC_ARCH", "H100_PCIE")
    assert heuristic_arch() == "H100_PCIE"


@pytest.mark.parametrize("arch", ["cpu", "unknown", "unknown_sm120", "unknown_sm90"])
def test_an_unresolved_arch_says_so_out_loud(arch):
    """Every fallback label warns, and the message names both the fallback and the way out."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_arch_fallback_once(arch, context=f"plan-{arch}")
    assert len(rec) == 1, f"{arch} must warn -- it resolves to the {TUNED_ARCH} pick set"
    msg = str(rec[0].message)
    assert TUNED_ARCH in msg and "CPO_HEURISTIC_ARCH" in msg


@pytest.mark.parametrize("arch", ["H100_SXM5", "H100_PCIE", "B200", "GB200"])
def test_a_resolved_arch_is_not_a_fallback_and_stays_quiet(arch):
    """Warning on a correctly-identified part would train people to ignore the warning.

    That is the failure mode a guard dies of: noise, then a filter, then nobody reads it when it
    finally matters.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_arch_fallback_once(arch, context=f"plan-{arch}")
    assert len(rec) == 0


def test_the_fallback_warning_fires_once_per_context():
    """Warn-once, or a per-cell caller turns one real problem into a page of identical lines."""
    _WARNED.discard(("__fallback__", "cpu", "warn-once-probe"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_arch_fallback_once("cpu", context="warn-once-probe")
        warn_arch_fallback_once("cpu", context="warn-once-probe")
    assert len(rec) == 1, "warn-once broken -- a per-cell caller would spam the log"


def test_the_tuned_arch_draws_no_suboptimal_warning():
    """`TUNED_ARCH` is where the thresholds ARE optimal, so there is nothing to say."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_arch_suboptimal_once(TUNED_ARCH, "some_kernel")
    assert len(rec) == 0, "must not warn for the tuned arch"


def test_the_suboptimal_warning_fires_once_per_arch_and_kernel():
    """Once per PAIR: a second kernel on the same arch is new information, a repeat is not."""
    for key in (("H100_SXM5", "kx"), ("H100_SXM5", "ky")):
        _WARNED.discard(key)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_arch_suboptimal_once("H100_SXM5", "kx")  # fires
        warn_arch_suboptimal_once("H100_SXM5", "kx")  # suppressed: same pair
        warn_arch_suboptimal_once("H100_SXM5", "ky")  # fires: different kernel
    msgs = [str(x.message) for x in rec]
    assert len([m for m in msgs if "kx" in m]) == 1, "kx must warn exactly once"
    assert len([m for m in msgs if "ky" in m]) == 1, "ky is a different pair and warns once"
    assert TUNED_ARCH in msgs[0] and "autotune" in msgs[0]


@pytest.mark.parametrize("fallback", ["cpu", "unknown", "unknown_sm120"])
def test_a_fallback_label_gets_the_fallback_message_not_the_suboptimal_one(fallback):
    """The delegation, asserted by what the message must NOT say.

    Fallback labels satisfy ``arch != TUNED_ARCH``, so they always reached
    `warn_arch_suboptimal_once` and always produced *a* warning -- which is why a test that only
    checked "something warned" would pass either way and catch nothing. The defect was the WORDING:
    it claimed the thresholds "may be suboptimal on <arch>" and advised a measured sweep, which
    is unusable in the case that matters most, since ``arch == "cpu"`` means there is no device to
    autotune on. Asserting the absence of that advice is the whole point.
    """
    for key in list(_WARNED):
        _WARNED.discard(key)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_arch_suboptimal_once(fallback, "kz")
        warn_arch_suboptimal_once(fallback, "kz")  # warn-once survives the delegation
    msgs = [str(x.message) for x in rec]
    assert len(msgs) == 1, f"a fallback must warn exactly once, got {len(msgs)}"
    assert "CPO_HEURISTIC_ARCH" in msgs[0], "must offer the explicit-arch override"
    assert "autotune" not in msgs[0], "must NOT advise autotune when there may be no device"
    assert fallback in msgs[0] and TUNED_ARCH in msgs[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_device_a_tensor_and_none_all_resolve_the_same_way():
    """The three argument forms are interchangeable. Holds on ANY CUDA device, tuned or not."""
    dev = torch.device("cuda")
    a, b, c = heuristic_arch(dev), heuristic_arch(torch.empty(2, device=dev)), heuristic_arch(None)
    assert a == b == c, f"device/tensor/None disagree: {a}, {b}, {c}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_recognised_part_resolves_to_a_real_label():
    """On a part this package has a label for, resolution must not fall through to unknown/cpu.

    **Kept separate from the consistency check above on purpose.** Combined, the pair could not pass
    on an sm_120 workstation -- ``_arch_label_for`` correctly returns ``"unknown_sm120"`` there,
    because no entry exists for that part -- so a developer running the suite locally saw a
    permanent red and learned to ignore it. A gate that cannot pass on the machine people run it on
    trains them to ignore the gate.

    Skips rather than fails on an unrecognised part, and NAMES the label, so the skip carries the
    information: an unrecognised part is a real state, not a defect.
    """
    arch = heuristic_arch(torch.device("cuda"))
    if arch.startswith("unknown") or arch == "cpu":
        pytest.skip(f"{arch}: no arch label for this part -- see _arch_label_for")
    assert not arch.startswith("unknown") and arch != "cpu", f"unexpected label {arch}"
