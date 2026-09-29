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
"""Tests for ``tests.distributed.topology`` -- the shared coupled-store topology guard.

**Every test here runs the real functions against a SYNTHETIC probe.** `build_p2p_table` is the one
thing this module does not own -- it is NVSHMEM's ``TEAM_SHARED`` translate, which requires a live
job and, measured, does not raise but hard-ABORTS the process (exit 255) when NVSHMEM is not up. So
the probe is monkeypatched and every p2p pattern is reachable in one ordinary launch, the same shape
`test_pe_map.py` uses. That is not a weakening: the guard's job is to turn a probe result into a
skip decision, and the probe result is exactly what is substituted.

**The load-bearing test is the ORDER one.** ``pe_table`` is an ORDERED, kernel-indexed table, and
the two comparisons that typecheck against an unordered roster -- set equality and ``sorted()`` --
are precisely the ones that discard the ordering that IS its content. Nothing in this module's own
logic reads the order, which is what makes a future "normalise the table first" edit look harmless;
`test_the_probe_receives_the_table_in_the_callers_order` is what makes it fail instead.
"""

import pytest

from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    matrix_exempt,
    no_unsupported,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt

from tests.distributed import topology

TOPOLOGY = KernelMatrix(
    kernel="topology",
    axes=(
        Axis(
            name="p2p_pattern",
            domain=(
                "any tuple of per-peer P2P-reachability booleans the store's probe can return, one "
                "entry per cp peer. The pool spans the two verdicts the guard must separate "
                "(all-True = a single NVLink domain, any-False = the mesh crosses one) at several "
                "cp, and deliberately places the False entry FIRST, LAST and INTERIOR: a guard "
                "written with an early-exit that inspects only one end passes on the other two "
                "positions, and a pool that only ever puts it last cannot tell those apart"
            ),
            values=(
                (True, True),  # cp=2, all-NVLink
                (True, False),  # cp=2, the peer is remote -- False LAST
                (False, True),  # cp=2, False FIRST
                (True, True, True, True),  # cp=4, all-NVLink
                (True, False, True, True),  # cp=4, False INTERIOR
                (True,) * 8,  # cp=8, the largest all-NVLink domain
                (True,) * 7 + (False,),  # cp=8, one remote peer at the boundary
                (False,) * 4,  # every peer remote -- a 4-node, 1-rank-per-node job
                (True,),  # cp=1: the table is this rank's own PE, always P2P with itself
            ),
            facets={
                "all_nvlink": lambda p: all(p),
                "has_ib": lambda p: not all(p),
                "false_first": lambda p: p[0] is False,
                "false_last": lambda p: p[-1] is False,
                "false_interior": lambda p: any(not v for v in p[1:-1]),
                "single_peer": lambda p: len(p) == 1,
                "large_cp": lambda p: len(p) >= 8,
            },
        ),
    ),
    computes=computes_nothing_numeric(
        because=(
            "the subject is a SKIP DECISION -- whether a coupled-store cell runs on this mesh. "
            "Nothing is launched and no tensor is produced, so there is no output whose element "
            "distribution could hide a defect. The failure this module guards against is a fault "
            "with no numbers at all: an illegal memory access on a NULL IB descriptor"
        )
    ),
    unsupported=no_unsupported(
        because=(
            "ENUMERATED, not sampled: the module has two public functions and NEITHER raises. "
            "job_has_ib_peers reduces the probe's output with all(); skip_if_coupled_cross_node "
            "either skips or returns. The one documented input requirement -- a non-empty table, "
            "since all(()) is True and would silently report 'no IB peers' -- is TRUSTED rather "
            "than checked, carried that way from the upstream, and is unreachable in practice "
            "because cp >= 1 always. There is no combination of declared p2p_pattern values that "
            "must raise, so a region would be a fiction"
        )
    ),
)


def _patch_probe(monkeypatch, pattern, seen=None):
    """Substitute the store's `build_p2p_table` with one returning `pattern`, recording its input.

    Args:
        monkeypatch: pytest's fixture. Required -- an unrestored patch would leak the fake probe
            into every later test in the session.
        pattern: The tuple of booleans the fake probe returns, regardless of its argument. Its
            length need not match the table's: the tests that vary them independently are asserting
            that the module reduces the PROBE's answer rather than re-deriving one from the table.
        seen: Optional list; the exact object the probe was called with is appended to it. This is
            how the ordering test observes the call, since the return value carries no trace of the
            argument.

    Returns:
        None. Patches `fold_cp_ops.distributed.gemm_sm90_a2a.build_p2p_table` for the test's
        duration.
    """
    import fold_cp_ops.distributed.gemm_sm90_a2a as G

    def fake(pe_table):
        if seen is not None:
            seen.append(pe_table)
        return pattern

    monkeypatch.setattr(G, "build_p2p_table", fake)


@TOPOLOGY.parametrize("p2p_pattern")
@numeric_exempt("asserts a topology VERDICT (bool), not a computed value")
def test_a_job_has_ib_peers_exactly_when_some_peer_is_not_p2p(monkeypatch, p2p_pattern):
    """``job_has_ib_peers`` is the negation of ``all(probe)`` -- at every declared pattern.

    Both directions matter and neither implies the other. A guard that never reports IB peers
    silently un-guards every coupled suite across nodes, which is the fault this module exists to
    prevent; a guard that always reports them skips every coupled cell on an ordinary single-node
    job, deleting the coverage instead.
    """
    _patch_probe(monkeypatch, p2p_pattern)
    table = tuple(range(len(p2p_pattern)))
    assert topology.job_has_ib_peers(table) is (not all(p2p_pattern)), (
        f"probe returned {p2p_pattern} (all={all(p2p_pattern)}) but job_has_ib_peers reported "
        f"{topology.job_has_ib_peers(table)}; the verdict must be the negation of all(probe)"
    )


@TOPOLOGY.parametrize("p2p_pattern")
@numeric_exempt("asserts whether a skip fired, not a computed value")
def test_a_coupled_cell_skips_exactly_on_a_mesh_that_crosses_nvlink(monkeypatch, p2p_pattern):
    """``skip_if_coupled_cross_node`` skips iff the probe reports a non-P2P peer.

    The skip is raised through `rank_invariant_skip`, so this also exercises the sanctioned path:
    a bare ``pytest.skip`` here would be a divergent-skip deadlock under ``tests/distributed/``, and
    the collective guard's runtime tripwire fails it as a FAIL rather than reporting it as a skip.
    """
    _patch_probe(monkeypatch, p2p_pattern)
    table = tuple(range(len(p2p_pattern)))
    expect_skip = not all(p2p_pattern)
    outcome = _run_capturing_skip(table)
    assert (outcome is not None) is expect_skip, (
        f"probe {p2p_pattern}: expected {'a skip' if expect_skip else 'no skip'}, got "
        f"{'skip: ' + str(outcome) if outcome else 'no skip'}"
    )
    if expect_skip:
        assert topology.COUPLED_NVLINK_ONLY_SKIP in outcome, (
            f"the skip fired but its reason {outcome!r} does not carry the standard text; a reason "
            "a reader cannot recognise is indistinguishable from an unrelated skip"
        )


def _run_capturing_skip(table, detail=""):
    """Call ``skip_if_coupled_cross_node`` and return its skip reason, or None if it did not skip.

    A test cannot let the skip propagate -- it would skip the test itself, and a skipped test
    asserts nothing. ``pytest.skip`` raises ``Skipped``, whose first argument is the reason.

    Args:
        table: The pe_table to pass through.
        detail: Forwarded to the guard. Optional.

    Returns:
        The reason string when the guard skipped, else None.
    """
    try:
        topology.skip_if_coupled_cross_node(table, detail=detail)
    except BaseException as e:  # noqa: BLE001 -- Skipped derives from BaseException, not Exception
        if type(e).__name__ != "Skipped":
            raise
        return e.args[0] if e.args else ""
    return None


@matrix_exempt(
    "the subject is the ARGUMENT the probe receives, which does not vary with the probe's return "
    "value -- the matrix axis is the one thing this test must hold fixed to observe the other"
)
@numeric_exempt("asserts an argument's identity and order, not a computed value")
def test_the_probe_receives_the_table_in_the_callers_order(monkeypatch):
    """The probe is called with the caller's table IN ORDER, not sorted, de-duplicated or set-ified.

    ``pe_table`` is ORDERED and kernel-indexed: entry ``r`` is the PE that cp-rank ``r`` maps to.
    Nothing in this module's own logic reads that order -- it only reduces the probe's answer with
    ``all()`` -- which is exactly what would make a future "normalise the table first" edit look
    free. It is not free: the probe resolves per-PE links, so a reordered table probes the same set
    of links but attributes them to the wrong ranks, and a scrambled mesh would be judged by
    another rank's connectivity.

    The pool is a SCRAMBLED table with no fixed point. Under the identity table this test cannot
    fail, because sorted, set-ified and original all coincide.
    """
    seen = []
    _patch_probe(monkeypatch, (True, True, True, True), seen=seen)
    scrambled = (3, 1, 0, 2)
    topology.job_has_ib_peers(scrambled)
    assert len(seen) == 1, f"expected exactly one probe call, saw {len(seen)}"
    assert tuple(seen[0]) == scrambled, (
        f"the probe received {tuple(seen[0])!r} but the caller passed {scrambled!r}. Order is the "
        "content of a pe_table: a set-equality or sorted() comparison typechecks here and discards "
        "exactly what makes the table a mapping"
    )


@matrix_exempt(
    "asserts that a non-int table is coerced, which is a property of the ARGUMENT's type rather "
    "than of the p2p pattern the matrix varies"
)
@numeric_exempt("asserts argument coercion, not a computed value")
def test_a_table_of_tensor_scalars_reaches_the_probe_as_plain_ints(monkeypatch):
    """A table whose entries are not ``int`` is coerced before the probe sees it.

    Callers pass ``pe_map.cp_pe_table.tolist()`` or iterate a tensor, so the entries arrive as
    numpy/torch scalars. NVSHMEM's binding takes a C int; handing it a wrapper is the kind of
    failure that surfaces as a type error deep in the FFI rather than at the call. The coercion is
    in this module, so it is asserted here.
    """
    import torch

    seen = []
    _patch_probe(monkeypatch, (True, True), seen=seen)
    topology.job_has_ib_peers(torch.tensor([1, 0]))
    assert all(type(p) is int for p in seen[0]), (
        f"the probe received {[type(p).__name__ for p in seen[0]]}; every entry must be a plain "
        "int, or the NVSHMEM binding sees a wrapper it cannot convert"
    )


@matrix_exempt("asserts the reason string's composition, which does not vary with the p2p pattern")
@numeric_exempt("asserts a message, not a computed value")
@pytest.mark.parametrize(
    "detail", ["", "back gemm-native store is a coupled store."], ids=["no_detail", "with_detail"]
)
def test_the_skip_reason_carries_the_callers_detail(monkeypatch, detail):
    """``detail`` is appended to the standard reason, and its absence leaves the reason unchanged.

    The detail is where a suite names the path that keeps the cross-node case covered, so a skip
    without it reads as a coverage hole rather than as a capability boundary with an alternative.
    """
    _patch_probe(monkeypatch, (True, False))
    reason = _run_capturing_skip((0, 1), detail=detail)
    assert reason is not None, "a probe reporting an IB peer must skip"
    assert reason.startswith(topology.COUPLED_NVLINK_ONLY_SKIP), (
        f"reason {reason!r} does not begin with the standard text"
    )
    assert (detail in reason) if detail else (reason == topology.COUPLED_NVLINK_ONLY_SKIP), (
        f"reason {reason!r} does not carry detail {detail!r}"
    )


@matrix_exempt(
    "a SOURCE-level invariant over the module's text; it holds for every p2p pattern at once and "
    "has no axis value to draw from"
)
@numeric_exempt("inspects source structure, not a computed value")
def test_the_module_owns_no_topology_knowledge_of_its_own():
    """Every topology fact comes from the store's probe -- this module holds no second copy.

    The module's whole argument is that it wraps `build_p2p_table` rather than re-implementing it,
    because a divergent copy would guard on a different topology than the kernel routes on. That is
    a claim about what the source does NOT contain, so it is checked here: the only NVSHMEM-facing
    name in the module is the probe's.

    Narrow on purpose. It cannot catch a copy written without any of these names -- a hand-rolled
    hostname comparison, say. It does catch the realistic drift, which is somebody reaching for the
    NVSHMEM team API directly when the probe's answer is inconvenient.

    **Matched over IDENTIFIERS, not over lines.** A line filter has to decide what counts as prose,
    and the first version of this test failed on its own subject: the module DOCSTRING names
    ``TEAM_SHARED`` while explaining which probe it delegates to, and a filter that skips lines
    containing ``\"\"\"`` skips the delimiters rather than the block between them. An AST walk never
    sees a docstring at all -- it is an `ast.Constant`, and only `ast.Name` and `ast.Attribute` are
    inspected -- so the module may describe the primitives it does not call.
    """
    import ast
    import inspect

    forbidden = {"team_translate_pe", "nvshmem_ptr", "TEAM_SHARED", "n_pes", "my_pe"}

    def offenders(src):
        """Identifiers from `forbidden` that `src` USES (docstrings and comments are invisible)."""
        used = {
            n.id if isinstance(n, ast.Name) else n.attr
            for n in ast.walk(ast.parse(src))
            if isinstance(n, (ast.Name, ast.Attribute))
        }
        return sorted(used & forbidden)

    # Positive control FIRST: a walk that inspects nothing reports a clean module just as loudly as
    # one that inspects everything, so prove it discriminates before believing its verdict. The two
    # halves are the two ways the real module could drift -- a bare call and an attribute call.
    control = "import nvshmem\ndef probe(t):\n    return nvshmem.core.team_translate_pe(t)\n"
    assert offenders(control) == ["team_translate_pe"], (
        f"the control module calls team_translate_pe and the walk reported {offenders(control)}; "
        "the check cannot be trusted against the real module until it fails against this one"
    )
    assert offenders("x = TEAM_SHARED\n") == ["TEAM_SHARED"], (
        "the walk does not see a bare Name reference, so a module reading TEAM_SHARED directly "
        "would pass"
    )

    hits = offenders(inspect.getsource(topology))
    assert not hits, (
        f"the module names NVSHMEM primitives {hits} in code rather than delegating to "
        "build_p2p_table. A second probe drifts from the store's, and then the guard and the "
        "kernel disagree on exactly the mesh where it matters"
    )
