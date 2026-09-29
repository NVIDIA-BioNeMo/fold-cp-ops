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

"""What a RUN actually exercised, as opposed to what a module parametrized.

Purpose
    `kernel_matrix.coverage_problems` asks whether a module PARAMETRIZES every facet it declares.
    For a mesh axis that question has a green answer that is false: one launch fixes one WORLD_SIZE,
    so a ``WORLD_SIZE=2`` run emits all 14 declared mesh specs and then SKIPS the 12 that need a
    different rank count. Statically every facet is hit; dynamically it exercised 2 of 14 and
    ``cross_node`` was never built -- measured, not estimated.
    Reporting that as "mesh fully covered" is worse than reporting nothing, because it is a claim.

    Failing the 2-rank run for it would be equally wrong -- the skip is correct, and one launch
    cannot cover a pool that spans world sizes. So coverage of such an axis is a property of the SET
    of launches, and this module is what makes that set inspectable: each run writes what it
    exercised, and the union across runs answers the question no single run can.

Semantics
    Two records per axis value, written by whoever knows which happened:

    * **exercised** -- the value was actually used (the mesh was built, the kernel ran).
    * **skipped** -- the value was parametrized and then declined, with the reason, so a hole is
      distinguishable from a value nobody ever wrote a test for.

    A session writes one JSON ledger next to its run report. `union_ledgers` merges every ledger in a
    directory; `unreached_facets` reports the facets no run in that set ever exercised.

    **Strict mode is opt-in and off by default** (`STRICT_ENV`). A single launch must never fail for
    its own world size -- that is the contradiction this module exists to avoid -- so the union check
    runs only where a sweep is actually being claimed, and is silent everywhere else.

Input requirements
    Values must be JSON-round-trippable to be compared across processes, and mesh specs are tuples,
    which JSON turns into lists. Everything is therefore keyed on `value_key`'s canonical string
    rather than on the object; a caller that invents its own key will silently fail to union with
    anybody else's ledger.

Raises
    Nothing on the recording path -- a ledger that raised would turn a diagnostic into an outage.
    `unreached_facets` returns problems; the caller decides whether they are fatal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

#: Env var that turns the union check into a failure. Off by default, deliberately: with it on by
#: default every ordinary ``torchrun --nproc_per_node=2`` run would fail for not having built a
#: 16-rank mesh, which is the false-negative twin of the false-positive this module fixes.
STRICT_ENV = "CPO_DIST_COVERAGE_STRICT"

#: Filename a session writes into the ledger directory. Carries the pid so concurrent ranks and
#: xdist workers do not clobber one another -- os.getpid() is NOT unique in some sandboxes, so the
#: caller passes a discriminator (see :func:`write_ledger`).
LEDGER_PREFIX = "coverage_ledger"

#: kernel -> axis -> {"exercised": {key: value}, "skipped": {key: [value, reason]}}
_LEDGER: Dict[str, Dict[str, Dict[str, dict]]] = {}


def value_key(value: Any) -> str:
    """Canonical, JSON-stable key for an axis value.

    Semantics
        A mesh spec is a tuple of ``(name, size)`` pairs where ``size`` may itself be a tuple. JSON
        has no tuple, so a naive round-trip turns those into lists and the same value stops matching
        itself across processes. Normalizing to a string BEFORE serializing is what lets two runs'
        ledgers union at all.

    Args:
        value: Any axis value. Tuples and lists are normalized to the same form recursively, so
            ``(("cp", 2),)`` and ``[["cp", 2]]`` share a key -- which they must, since one is what
            the other becomes after a JSON round trip.

    Returns:
        A deterministic string. Not intended to be parsed back; it is an identity, not a
        serialization.
    """

    def norm(v):
        if isinstance(v, (tuple, list)):
            return [norm(x) for x in v]
        return v

    return json.dumps(norm(value), sort_keys=True, default=str)


def record_exercised(kernel: str, axis: str, value: Any) -> None:
    """Note that a run actually USED an axis value.

    Args:
        kernel: The matrix's ``kernel`` name, matching `KernelMatrix.kernel`.
        axis: Axis name, e.g. ``"mesh"``.
        value: The value used. Keyed via :func:`value_key`, so any JSON-able shape works.

    Returns:
        None. Recording an already-recorded value is a no-op, so a value used by twenty tests
        appears once.
    """
    slot = _LEDGER.setdefault(kernel, {}).setdefault(axis, {"exercised": {}, "skipped": {}})
    slot["exercised"][value_key(value)] = value
    slot["skipped"].pop(value_key(value), None)


def record_skipped(kernel: str, axis: str, value: Any, reason: str) -> None:
    """Note that a run PARAMETRIZED an axis value and then declined it.

    Semantics
        Recorded with the reason so the ledger distinguishes "this launch could not run it" from
        "nobody wrote a test". A value already recorded as exercised is NOT downgraded -- one test
        skipping it while another used it still means the run exercised it.

    Args:
        kernel: The matrix's ``kernel`` name.
        axis: Axis name.
        value: The declined value.
        reason: Why, verbatim -- the skip message is the right thing to pass, since it already names
            the rank count or the topology that was missing.

    Returns:
        None.
    """
    slot = _LEDGER.setdefault(kernel, {}).setdefault(axis, {"exercised": {}, "skipped": {}})
    key = value_key(value)
    if key in slot["exercised"]:
        return
    slot["skipped"][key] = [value, reason]


def snapshot() -> Dict[str, Dict[str, Dict[str, dict]]]:
    """The current in-process ledger.

    Returns:
        A deep-ish copy safe to serialize; mutating it does not affect further recording.
    """
    return json.loads(json.dumps(_LEDGER, default=str))


def reset() -> None:
    """Clear the in-process ledger, for tests of this module and for a fresh session.

    Returns:
        None.
    """
    _LEDGER.clear()


def ledger_dir(default: Optional[Path] = None) -> Optional[Path]:
    """Where ledgers are written and unioned from.

    Args:
        default: Directory to use when the env var is unset -- the caller passes the run-report
            directory, so a ledger lands beside the report a reader is already looking at.

    Returns:
        The directory, or ``None`` when neither an env var nor a default is available (a session
        whose basetemp could not be resolved). Never creates it; :func:`write_ledger` does.
    """
    env = os.environ.get("CPO_DIST_LEDGER_DIR", "").strip()
    if env:
        return Path(env)
    return default


def write_ledger(directory, discriminator: str) -> Optional[Path]:
    """Write this session's ledger as JSON.

    Args:
        directory: Destination; created if absent. ``None`` writes nothing and returns ``None``, so
            a session with no resolvable temp dir degrades to silence rather than an error.
        discriminator: Made part of the filename to keep concurrent writers apart. Pass something
            genuinely unique -- a pid is NOT sufficient everywhere (measured: every backgrounded
            process in this sandbox reports pid 25), so callers combine rank, worker id and a random
            or time-derived component.

    Returns:
        The path written, or ``None`` if there was nothing to write or no directory. An empty ledger
        writes nothing: a file saying "this run exercised nothing" would union harmlessly but makes
        the directory listing lie about how many real runs are in it.
    """
    if directory is None or not _LEDGER:
        return None
    d = Path(directory)
    try:
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{LEDGER_PREFIX}.{discriminator}.json"
        path.write_text(json.dumps(_LEDGER, indent=2, default=str))
        return path
    except OSError:
        return None  # a diagnostic must never take a session down


def union_ledgers(directory) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """Merge every ledger in a directory into one view.

    Semantics
        Exercised wins over skipped across files, which is the whole point: the ``WORLD_SIZE=2`` run
        skipped ``cp=16`` and the 16-rank run exercised it, so the union says exercised. A value
        skipped by every run stays skipped and is what `unreached_facets` reports on.

    Args:
        directory: Directory of ``coverage_ledger.*.json`` files. A missing or unreadable directory
            yields ``{}``, and a corrupt individual file is skipped rather than raising -- a
            half-written ledger from a killed run must not break the check for the others.

    Returns:
        The merged ledger, same shape as :func:`snapshot`.
    """
    merged: Dict[str, Dict[str, Dict[str, dict]]] = {}
    if directory is None:
        return merged
    d = Path(directory)
    if not d.is_dir():
        return merged
    for f in sorted(d.glob(f"{LEDGER_PREFIX}.*.json")):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for kernel, axes in data.items():
            for axis, rec in axes.items():
                slot = merged.setdefault(kernel, {}).setdefault(
                    axis, {"exercised": {}, "skipped": {}}
                )
                slot["exercised"].update(rec.get("exercised", {}))
                slot["skipped"].update(rec.get("skipped", {}))
    for axes in merged.values():
        for slot in axes.values():
            for key in list(slot["exercised"]):
                slot["skipped"].pop(key, None)
    return merged


def unreached_facets(matrix, merged: Mapping[str, Any]) -> List[str]:
    """Facets that no run in the union ever EXERCISED.

    Semantics
        The dynamic counterpart to `kernel_matrix.coverage_problems`. That one asks whether the
        module parametrized a facet; this asks whether any launch actually reached it. A facet the
        module never parametrizes is the static check's business and is not reported twice here.

        A facet's `Axis.waived` entry suppresses it, exactly as in the static check -- a waiver is
        prose the reader can weigh, and having to write it in two places would make the two drift.

    Args:
        matrix: The `KernelMatrix` whose axes and facets define what "covered" means.
        merged: Output of :func:`union_ledgers`.

    Returns:
        One message per unreached facet, naming the axis, the facet, and the values that were
        skipped and by what reason -- so the reader learns WHICH launch is missing, not merely that
        one is. Empty when every facet was reached, when the kernel has no ledger at all (nothing
        ran, which is not this check's business to diagnose), or when every gap is waived.
    """
    axes = merged.get(matrix.kernel)
    if not axes:
        return []
    problems: List[str] = []
    for axis in matrix.axes:
        rec = axes.get(axis.name)
        if not rec:
            continue
        exercised = list(rec.get("exercised", {}).values())
        skipped = rec.get("skipped", {})
        for fname, pred in axis.facets.items():
            if axis.waived.get(fname):
                continue
            if any(_safe_pred(pred, v) for v in exercised):
                continue
            blockers = {reason for _value, reason in skipped.values() if _safe_pred(pred, _value)}
            if not blockers:
                continue  # nothing was skipped for it either -> the STATIC check owns this
            problems.append(
                f"{matrix.kernel}: axis {axis.name!r} facet {fname!r} was parametrized but NO run "
                f"in this ledger set ever exercised it. Every value carrying it was skipped: "
                + "; ".join(sorted(blockers))
                + f". Add the launch that can run it, or waive the facet on the {axis.name!r} axis "
                f"with a written reason."
            )
    return problems


def _safe_pred(pred, value) -> bool:
    """Apply a facet predicate to a value that has been through JSON, without raising.

    Semantics
        Facet predicates are written against the DECLARED value -- tuples of tuples for a mesh spec
        -- while a value read back from a ledger is lists of lists. This retries the predicate on a
        tuple-ized form, and treats any failure as "does not carry the facet" rather than letting a
        diagnostic raise.

    Args:
        pred: The facet predicate from `Axis.facets`.
        value: A value from the ledger, post-JSON.

    Returns:
        The predicate's verdict, or ``False`` if it could not be applied.
    """

    def tuplize(v):
        if isinstance(v, list):
            return tuple(tuplize(x) for x in v)
        return v

    for candidate in (tuplize(value), value):
        try:
            return bool(pred(candidate))
        except Exception:  # noqa: BLE001 - a facet a value cannot answer is simply not carried
            continue
    return False


def strict_enabled() -> bool:
    """Whether the union check should FAIL the session rather than just report.

    Returns:
        ``True`` when `STRICT_ENV` is set to something other than ``"0"``/empty. Off by default so an
        ordinary single-world-size launch is never failed for its own topology.
    """
    return os.environ.get(STRICT_ENV, "").strip() not in ("", "0")


def format_summary(merged: Mapping[str, Any]) -> Iterable[str]:
    """Human-readable exercised/skipped counts, for the terminal summary.

    Args:
        merged: Output of :func:`union_ledgers`, or a single session's :func:`snapshot`.

    Returns:
        One line per (kernel, axis), sorted, e.g.
        ``distributed_manager.mesh: exercised 4, skipped 11``. Empty when nothing was recorded, so a
        session with no distributed tests prints nothing.
    """
    lines: List[str] = []
    for kernel in sorted(merged):
        for axis in sorted(merged[kernel]):
            rec = merged[kernel][axis]
            lines.append(
                f"{kernel}.{axis}: exercised {len(rec.get('exercised', {}))}, "
                f"skipped {len(rec.get('skipped', {}))}"
            )
    return lines
