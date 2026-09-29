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

"""Arch-awareness for the size->config heuristics, and for those ONLY.

A size->config heuristic -- a `select="heuristic"` formula that picks a tile or a schedule from the
shapes alone, without timing anything -- splits into two layers that look alike in the code and are
not alike at all:

* a **validity** layer (``k % 8``, ``2N % tile_N == 0``, a pingpong-legal ``tile_M``): hardware
  CORRECTNESS gates. A tile the kernel cannot launch cannot launch on any SM90 part, so these are
  arch-INDEPENDENT and stay un-branched.
* a **perf-regime** layer (the bound-regime thresholds and bisected crossovers): these were derived
  by measurement on ONE part, and there is no reason to expect the crossover to sit at the same
  shape on another.

This module makes only the second layer arch-aware. A heuristic asks `heuristic_arch` for a label,
warns ONCE through `warn_arch_suboptimal_once` if that label is not `TUNED_ARCH`, and then indexes
an arch-keyed table -- so supporting a new part is a new dict entry rather than a code fork, and a
threshold that was never validated on the running hardware SAYS so instead of being silently
believed.

**The failure this exists to prevent is a silent one.** `heuristic_arch()` reads the live device and
has no other signal, so a plan computed off the target hardware falls back to `TUNED_ARCH` and looks
exactly like a plan computed on it. Set ``CPO_HEURISTIC_ARCH`` to make an off-device pick explicit.
"""

import os
import warnings
from functools import lru_cache
from typing import Union

import torch
from torch import Tensor

__all__ = ["heuristic_arch", "TUNED_ARCH", "warn_arch_fallback_once", "warn_arch_suboptimal_once"]

#: The arch every shipped perf threshold was bisected on. A heuristic running anywhere else uses
#: THIS arch's numbers and warns once; see `warn_arch_suboptimal_once`.
TUNED_ARCH = "H200_SXM5"


@lru_cache(maxsize=None)
def _arch_label_for(name: str, cc: int, sm_count: int) -> str:
    """Map a device's (name, packed capability, SM count) to an arch label.

    Purpose
        The one place a marketing name becomes a key an arch-keyed threshold table can be indexed
        by. Split out from `heuristic_arch` so it is a pure function of three scalars -- which is
        what makes it cacheable, and testable with no CUDA device present.

    Semantics
        The device NAME is the primary signal; the SM count only disambiguates the SXM5-vs-PCIe
        split where the name alone does not carry the form factor. **Match order is load-bearing
        and not alphabetical:** "GH200" contains "H200" as a substring and "GB200" contains "B200",
        so the superchip and export parts are tested BEFORE the substring they contain. Reordering
        these labels a Grace-Hopper part as the tuned arch and hands it thresholds bisected on a
        different machine.

        An unrecognised part returns ``"unknown_sm{cc}"`` rather than raising or guessing: a label
        that names the capability is enough for `warn_arch_fallback_once` to say something true,
        and a guess would be indistinguishable from a real match.

    Args:
        name: ``torch.cuda.get_device_name()``. Matched case-insensitively; may be any string,
            including one with no recognised substring.
        cc: The compute capability PACKED as ``major * 10 + minor`` -- 90 for SM90, not 9. Passing
            the major alone silently falls through every branch to ``"unknown_sm9"``, which is a
            label no threshold table has an entry for, so the heuristic quietly uses the fallback.
        sm_count: ``multiprocessor_count``, or -1 when unavailable. Only consulted for the H100
            PCIe-vs-SXM5 split, so a wrong value elsewhere costs nothing.

    Returns:
        One of the known labels (``"H200_SXM5"``, ``"H100_SXM5"``, ``"H100_PCIE"``, ``"GH200"``,
        ``"H800"``, ``"B200"``, ``"GB200"``, ``"B100"``) or ``f"unknown_sm{cc}"``.
    """
    n = name.upper()
    # ── Hopper (SM90 = cc 90) ──
    # Order matters: "GH200" contains "H200", so the Grace-Hopper / export parts come FIRST.
    if cc == 90:
        if "GH200" in n:
            return "GH200"  # Grace-Hopper superchip (SM90 Hopper GPU)
        if "H800" in n:
            return "H800"  # the export-market Hopper part
        if "H200" in n:
            # H200 ships only as the SXM5 part (132 SMs, 141 GB HBM3e) and its device name carries
            # no "SXM" substring, so the H200 name match IS the SXM5 signal. This is the TUNED arch.
            return "H200_SXM5"
        if "H100" in n:
            # H100 comes as SXM5 (132 SMs) and PCIe (114 SMs). The name usually carries the form;
            # the SM count decides when it does not.
            if "PCIE" in n or sm_count == 114:
                return "H100_PCIE"
            return "H100_SXM5"
        return f"unknown_sm{cc}"
    # ── Blackwell (SM100 = cc 100) ── same substring trap: GB200 before B200.
    if cc == 100:
        if "GB200" in n:
            return "GB200"
        if "B200" in n:
            return "B200"
        if "B100" in n:
            return "B100"
        return f"unknown_sm{cc}"
    # everything else (Ampere, Ada, a future part): a capability-keyed label, never a guess.
    return f"unknown_sm{cc}"


def heuristic_arch(device_or_tensor: Union[torch.device, Tensor, None] = None) -> str:
    """The arch label a size->config heuristic should key its perf thresholds on.

    Purpose
        Answers "which machine's measurements apply here?" so a threshold table can be indexed
        instead of hardcoded.

    Semantics
        Reads the live device through torch and maps it with `_arch_label_for`, which is memoized --
        so repeated calls on a hot path cost one `torch.cuda` query, not a string match.

        ``CPO_HEURISTIC_ARCH`` overrides the detection entirely and is consulted FIRST, before the
        device is even looked at. That is for computing a plan for hardware you are not running on:
        without it such a plan silently reports `TUNED_ARCH`'s picks, which is the failure mode this
        module's docstring opens with.

        Every failure to resolve returns a LABEL rather than raising. A heuristic must still return
        a config on a machine with no CUDA device (an offline audit, a CPU-only CI job), and the
        caller distinguishes the cases by the label: ``"cpu"``, ``"unknown"`` (torch raised) or
        ``"unknown_sm{cc}"`` (a real device, unrecognised part).

    Args:
        device_or_tensor: A `torch.device`, a `Tensor` whose ``.device`` is used, or None for the
            current device. **Passing None from a measurement harness is the known footgun**: the
            result then depends on ambient CUDA state rather than on the tensors being timed, so a
            recorded pick label can disagree with the kernel that was actually timed. Pass
            ``x.device``.

    Returns:
        An arch label string, always. Never raises.
    """
    # The EXPLICIT off-device override, checked before anything reads the device.
    override = os.environ.get("CPO_HEURISTIC_ARCH", "").strip()
    if override:
        return override
    device = device_or_tensor.device if isinstance(device_or_tensor, Tensor) else device_or_tensor
    if not torch.cuda.is_available():
        return "cpu"
    try:
        name = torch.cuda.get_device_name(device)
        cap = torch.cuda.get_device_capability(device)
        props = torch.cuda.get_device_properties(device)
        sm_count = getattr(props, "multi_processor_count", -1)
    except Exception:
        return "unknown"
    return _arch_label_for(name, cap[0] * 10 + cap[1], sm_count)


#: ``(arch, kernel)`` pairs already warned about, so a warning fires once per process rather than
#: once per launch. Module-level, hence shared by every heuristic -- which is the point.
_WARNED: set = set()


def warn_arch_fallback_once(arch: str, context: str = "heuristic plan") -> None:
    """Warn ONCE that a pick was computed with `TUNED_ARCH`'s numbers because no arch was resolved.

    Purpose
        Distinguishes "this machine is not the tuned one" from "no machine was identified at all".
        Only the second is silently wrong in both directions: the label reported and the thresholds
        used both come from a fallback, so a plan computed on a laptop reads exactly like one
        computed on the target.

    Semantics
        A no-op unless `arch` is a fallback label (``"cpu"``, ``"unknown"``, ``"unknown_sm*"``), and
        a no-op the second time for the same ``(arch, context)`` pair. Warns through
        `warnings.warn` at ``stacklevel=3``, so the message points at the heuristic's CALLER rather
        than at this module.

        **It does not currently change any answer**, only because the shipped ``H200_SXM5`` and
        ``H100_SXM5`` entries happen to hold the same numbers. That is a property of today's table,
        not a guarantee -- the moment one entry is re-baked, every off-device plan becomes wrong.

    Args:
        arch: A label from `heuristic_arch`. A real arch label makes this a no-op, so it is safe to
            call unconditionally.
        context: What was being computed, interpolated into the message. Also part of the
            warn-once key, so two different plans each get one warning.

    Returns:
        None. Emits a `UserWarning` as a side effect.
    """
    if arch not in ("cpu", "unknown") and not arch.startswith("unknown_sm"):
        return
    key = ("__fallback__", arch, context)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"fold_cp_ops heuristic_arch() resolved {arch!r} -- the {context} is being computed with "
        f"the {TUNED_ARCH} pick set, NOT the target arch's. Set CPO_HEURISTIC_ARCH=<label> to "
        f"select an arch explicitly, or compute the plan on the target device.",
        stacklevel=3,
    )


def warn_arch_suboptimal_once(arch: str, kernel_name: str) -> None:
    """Warn ONCE that a kernel's perf thresholds were bisected on `TUNED_ARCH`, not on `arch`.

    Purpose
        The one call every arch-keyed heuristic makes, so "this threshold is ported, not validated
        here" is stated at the point of use instead of living in a comment nobody reads.

    Semantics
        Silent on `TUNED_ARCH` -- that is where the numbers ARE optimal -- and silent the second
        time for a given ``(arch, kernel_name)`` pair.

        A FALLBACK label is DELEGATED to `warn_arch_fallback_once` rather than warned about here.
        Fallback labels are a strict subset of ``arch != TUNED_ARCH`` so they reach this function
        anyway, but the suboptimal wording misdescribes them: the problem is not "these thresholds
        may be suboptimal on this arch", it is "no arch was identified". The remedy this message
        suggests is also unusable in exactly that case, since measuring needs a device to measure
        on. Delegating here wires the fallback warning into every call site without touching any of
        them.

        **The remedy this warning names is THIS package's, not the upstream's.** The upstream spells
        the measured path ``select="autotune"``; here it is a ``do_autotune=True`` gate on the
        kernel's own entry. A warning that advises an argument the package does not accept is worse
        than no warning -- it sends the one reader who acts on it to a ``TypeError``.

    Args:
        arch: A label from `heuristic_arch`.
        kernel_name: The kernel whose heuristic is speaking, interpolated into the message and part
            of the warn-once key. Use the public entry point's name -- it is what the reader would
            grep for.

    Returns:
        None. Emits a `UserWarning` as a side effect.
    """
    if arch == TUNED_ARCH:
        return
    if arch in ("cpu", "unknown") or arch.startswith("unknown_sm"):
        warn_arch_fallback_once(arch, f"{kernel_name} heuristic plan")
        return
    key = (arch, kernel_name)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"fold_cp_ops {kernel_name} size->config heuristic is tuned for {TUNED_ARCH}; on {arch} "
        f"the perf thresholds may be suboptimal -- pass do_autotune=True (and freeze the "
        f"winner) for the true per-shape optimum.",
        stacklevel=3,
    )
