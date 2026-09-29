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

"""One declared test matrix per kernel, so a config cannot be forgotten by accident.

**The problem this exists to stop.** Every test in this suite used to carry its own hand-written
``@pytest.mark.parametrize`` list. Nothing related those lists, so a shape class could be absent
from all of them and look exactly like a shape class that had been considered and excluded. That is
not hypothetical: bf16 odd N, the TriMul feature dims 128 and 384, and the whole
``threads_per_row=8`` rung were each missing from the LayerNorm grid for exactly that reason, and
each was found by hand rather than by the suite.

**The shape of the fix is four objects and two rules.**

* :class:`Axis` — one parametrized variable. It carries the ``domain`` the kernel *claims* to
  accept, the ``values`` actually in the pool, and ``facets``: named predicates that a diverse pool
  should land on both sides of.
* :class:`KernelMatrix` — a named bundle of axes, and the only sanctioned source of test values.
* :class:`Unsupported` — a combo the kernel must **refuse**, at its own front door. Declaring one
  obliges a guard; see :data:`API_LEVEL_ERRORS` for why it cannot be satisfied by an internal
  explosion or by an ``assert``.
* :func:`matrix_exempt` — the escape hatch, which **requires a written reason**.

Rule one: a test either parametrizes from the matrix or is exempt with a reason. Rule two: every
matrix states what it refuses -- ``unsupported=`` is mandatory, and ``no_unsupported(because=...)``
is how a kernel says there is nothing. Both are enforced by ``audit_test_module``, which
``tests/conftest.py`` runs over every collected module and
``tests/testing/test_kernel_matrix.py`` runs again as a test.

**Restricting is allowed; restricting silently is not.** A test that needs a subset passes
``only=``/``drop=`` *and* ``because="..."``. That keeps each test as cheap as it is today while
making every exclusion a sentence somebody had to write. Values outside the pool are rejected
outright — otherwise a test could invent shapes the diversity check never sees, and the guarantee
would be hollow.

**The diversity check.** For each facet, ``p`` is the fraction of pool values satisfying it and the
score is the binary entropy ``H(p)``: 0 when every value falls on one side, 1 when they are evenly
split. Requiring ``H >= threshold`` is what catches "the domain says any integer but the pool is all
powers of two" -- there ``p == 1`` for the ``power_of_two`` facet, so ``H == 0``. A facet that is
genuinely unreachable is waived *with a reason*, never deleted.

Deliberately not here: value generation, random sampling, shrinking. The pool is written down and
reviewed. This module only stops it from being quietly bypassed.
"""

import ast
import contextlib
import pathlib
import sys
import math
import traceback as _traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

import pytest

#: Minimum binary entropy a facet must reach before the pool counts as covering it. 0.25 admits a
#: 1-in-20 minority (H(0.05) = 0.286) and rejects 0-in-N and N-in-N, which are the degenerate cases
#: worth failing on. Raising it demands a more even split, not merely a token representative.
MIN_FACET_ENTROPY = 0.25

#: Every matrix built in this process, keyed by kernel name. The diversity and lock tests iterate
#: it, so a matrix only participates once the module defining it has been imported.
REGISTRY: Dict[str, "KernelMatrix"] = {}


#: Which axis values each TEST MODULE actually emitted, ``kernel -> module -> axis -> {values}``.
#:
#: **This is what makes a declared axis more than decoration.** `Axis.diversity` scores the declared
#: POOL, never what ran, so a pool wide enough to satisfy the entropy floor can still be narrowed to
#: one value by every test that uses it -- ``only={"K": (128,)}, because="compile cost"`` is fully
#: compliant and covers nothing. Recording emissions lets the audit ask the question diversity
#: cannot: did this module, ACROSS ALL ITS TESTS, ever run the large shape?
#:
#: Keyed by module so a kernel's correctness gate and its perf gate are judged separately -- the
#: perf gate legitimately pins tiny cells (dispatch is shape-independent, so its 8x8x16 measures the
#: submit path and a large shape would measure the kernel instead), and merging the two would let
#: either borrow the other's coverage.
_EMITTED: Dict[str, Dict[str, Dict[str, set]]] = {}

#: Cells whose axis values land in MORE THAN ONE declared `Unsupported` region, keyed by kernel.
#:
#: Overlap is legal -- :meth:`KernelMatrix.parametrize_unsupported` admits every matching region's
#: error -- but it is worth SEEING, for two reasons. It is a signal the regions may be poorly
#: factored (a cell that violates three constraints usually wants one of them declared elsewhere),
#: and it is the population a reviewer needs when judging whether an alternation has become so wide
#: that the assertion no longer says much.
#:
#: **It is recorded rather than raised, and that is the correction to how this was found.** The
#: defect it replaces -- first-match-wins, silently -- produced 19 uncaught exceptions and a
#: 16-rank desync, and it was statically visible the whole time: 36 of 284 emitted cells in one
#: matrix. Nothing looked, because nothing was asked to. What must never be silent again is the
#: INVARIANT (every matching region admitted), which is asserted at the emission site; the overlap
#: itself is information, and information belongs in a place a reader can query.
_UNSUPPORTED_OVERLAP: Dict[str, list] = {}


def _record_overlap(kernel: str, names: Sequence[str], cells: Sequence[Any]) -> None:
    """Record the cells that matched several `Unsupported` regions, for later inspection.

    Args:
        kernel: The matrix's kernel name.
        names: Axis names, positionally matching each recorded row.
        cells: ``(row, (raises_name, ...))`` pairs, one per overlapping cell.

    Returns:
        None; appends to `_UNSUPPORTED_OVERLAP` in place. Never raises: this is a reporting
        channel, and a reporting channel that can fail the run it reports on is worse than none.
    """
    _UNSUPPORTED_OVERLAP.setdefault(kernel, []).append(
        {"names": tuple(names), "cells": list(cells)}
    )


def unsupported_overlap(kernel: str | None = None):
    """The recorded multi-region cells, for a kernel or for every kernel seen so far.

    Args:
        kernel: A kernel name, or None for the whole mapping. An unknown name yields an empty list
            rather than raising -- absence of overlap and absence of a sweep are both "nothing to
            report", and distinguishing them would invite a caller to treat one as a failure.

    Returns:
        A list of ``{"names": (...), "cells": [(row, (raises_name, ...)), ...]}`` records for one
        kernel, or the full ``{kernel: [record, ...]}`` mapping when ``kernel`` is None. Populated
        as a side effect of :meth:`KernelMatrix.parametrize_unsupported`, so it is only as complete
        as the sweeps that have been collected in this process.
    """
    return _UNSUPPORTED_OVERLAP if kernel is None else _UNSUPPORTED_OVERLAP.get(kernel, [])


def _record_emission(kernel: str, names: Sequence[str], grid: Sequence[Sequence[Any]]) -> None:
    """Record the axis values one ``parametrize`` call emitted, against its calling module.

    Args:
        kernel: The matrix's kernel name.
        names: Axis names, positionally matching each row of `grid`.
        grid: The emitted rows.

    Returns:
        None; updates `_EMITTED` in place. Unhashable values are skipped rather than raising -- a
        pool of tensors would be a different problem, and losing coverage credit is the safe way to
        fail here.
    """
    mod = sys._getframe(2).f_globals.get("__name__", "?")
    per_axis = _EMITTED.setdefault(kernel, {}).setdefault(mod, {})
    for row in grid:
        for name, value in zip(names, row):
            try:
                per_axis.setdefault(name, set()).add(value)
            except TypeError:  # pragma: no cover - an unhashable pool value
                pass


def binary_entropy(p: float) -> float:
    """Shannon entropy of a Bernoulli(``p``) variable, in bits.

    Args:
        p: Fraction in ``[0, 1]``. Values outside that range are a caller bug and raise, because a
            silently clamped fraction would report a coverage score that does not correspond to the
            pool it was computed from.

    Returns:
        ``0.0`` at ``p in {0, 1}`` (the pool is entirely on one side of the facet, i.e. no
        coverage), rising to ``1.0`` at ``p == 0.5`` (evenly split).

    Raises:
        ValueError: If ``p`` is outside ``[0, 1]``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be a fraction in [0, 1], got {p!r}")
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


@dataclass(frozen=True)
class FacetScore:
    """How well one facet is covered by one axis's pool. Returned by :meth:`Axis.diversity`.

    Attributes:
        axis: Name of the axis scored.
        facet: Name of the predicate scored.
        hits: How many pool values satisfy the predicate.
        total: Pool size. Always positive -- :class:`Axis` rejects an empty pool.
        entropy: ``binary_entropy(hits / total)``; 0 means the pool is entirely on one side.
        waived: Reason this facet is exempt from the threshold, or None if it must meet it.
    """

    axis: str
    facet: str
    hits: int
    total: int
    entropy: float
    waived: str | None = None

    def __str__(self) -> str:
        """One-line rendering used verbatim in assertion messages and coverage reports.

        Returns:
            A line like ``N.odd  6/35  H=0.66``, suffixed with the waiver reason when waived.
        """
        base = f"{self.axis}.{self.facet}  {self.hits}/{self.total}  H={self.entropy:.2f}"
        return f"{base}  [waived: {self.waived}]" if self.waived else base


@dataclass(frozen=True)
class Axis:
    """One parametrized variable: what the kernel claims to accept, and what is actually tested.

    Attributes:
        name: The pytest argument name. Must match the test function's parameter exactly.
        domain: Human-readable statement of what the kernel accepts along this axis, e.g. ``"any
            positive int; only constraint is the 16-byte alignment floor"``. Never parsed -- it is
            printed in failure messages so a reader can judge whether the pool is honest about the
            claim. Write the claim, not the pool.
        values: The pool. Every test drawing on this axis selects from these and may not add to
            them. Must be non-empty and free of duplicates; duplicates would skew every facet
            fraction and silently weaken the diversity check.
        facets: Named predicates over a value. A pool is expected to land on **both** sides of each
            one. Keep them cheap and total -- a facet that raises on a legal value fails collection,
            not the test.
        waived: Facet name -> why the pool cannot cover it (e.g. a dtype the DSL cannot compile).
            Exempts that facet from the threshold while keeping it visible in the report. A waiver
            for an unknown facet name is a typo and raises; so does an empty reason, since a
            reasonless waiver is indistinguishable from deleting the check.
    """

    name: str
    domain: str
    values: Tuple[Any, ...]
    facets: Mapping[str, Callable[[Any], bool]] = field(default_factory=dict)
    waived: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject the pools that would make the diversity check meaningless.

        Returns:
            None.

        Raises:
            ValueError: If the pool is empty, contains duplicates, or a waiver names a facet that
                does not exist.
        """
        if not self.values:
            raise ValueError(f"axis {self.name!r}: pool is empty")
        seen = []
        for v in self.values:
            if any(v is s or v == s for s in seen):
                raise ValueError(f"axis {self.name!r}: duplicate value {v!r} skews facet fractions")
            seen.append(v)
        unknown = set(self.waived) - set(self.facets)
        if unknown:
            raise ValueError(f"axis {self.name!r}: waiver for unknown facet(s) {sorted(unknown)}")
        blank = sorted(f for f, why in self.waived.items() if not (why and why.strip()))
        if blank:
            raise ValueError(
                f"axis {self.name!r}: waived facet(s) {blank} give no reason. A waiver suppresses "
                f"a coverage failure, so it has to say why the facet is unreachable -- otherwise "
                f"it is indistinguishable from deleting the check."
            )

    def diversity(self) -> list[FacetScore]:
        """Score every facet against the pool.

        Returns:
            One :class:`FacetScore` per facet, in declaration order. An axis with no facets returns
            an empty list, which the checker treats as "nothing claimed, nothing to verify".
        """
        n = len(self.values)
        return [
            FacetScore(
                axis=self.name,
                facet=fname,
                hits=(hits := sum(1 for v in self.values if pred(v))),
                total=n,
                entropy=binary_entropy(hits / n),
                waived=self.waived.get(fname),
            )
            for fname, pred in self.facets.items()
        ]


#: The two spellings of a token extent. Values are what a test parametrizes; the kernel side is
#: whether the extent reaches the compiler as a baked constant or as a marked runtime value.
SHAPE_MODES = ("static", "dynamic")


def shape_mode_axis(*, waived: Mapping[str, str] | None = None) -> Axis:
    """The STATIC-vs-DYNAMIC token-extent axis, declared once so every module spells it the same.

    Purpose
        A kernel compiled with a BAKED token extent and the same kernel compiled with that extent
        MARKED runtime are different generated code, not the same code run twice. Constant folding
        is the difference: a stride that is a compile-time constant participates in compile-time
        arithmetic, and a stride that is a runtime value does not. Without this axis the two are
        indistinguishable in the matrix, so "we never tested static at this size" and "static is
        fine at this size" are the same state in the source -- which is the exact blind spot the
        matrix convention exists to remove.

    Why it is declared here and not per module
        The name is load-bearing: `coverage_problems` keys on it, and a module that spelled it
        ``shape_kind`` would satisfy nothing while looking compliant. A shared constructor makes the
        name unspellable-wrong.

    Why BOTH facets, when the pool has exactly two values
        The facets are deliberately complementary. `coverage_problems` fails a facet that no emitted
        value satisfies, so with both declared a module that only ever runs dynamic fails
        ``static_extent`` and a module that only ever runs static fails ``dynamic_extent``. A single
        ``dynamic_extent`` facet would be satisfied by a dynamic-only module -- i.e. it would pass in
        precisely the case this axis exists to catch.

    MEASURED motivation (2026-08-19)
        The front-A2A IB drain formed ``dst_feat * epi_n * M_full`` in 32 bits and wrapped once
        ``(2*Dloc - epi_n) * M_full >= 2**31``, producing a CUDA illegal address. It reproduced ONLY
        with a static extent: marking the recv token dim made the stride a runtime value and the
        wrap disappeared. The shipped profiling harness hardcodes ``--dynamic``, so 23 archived nsys
        cells never touched it, and **45 of 128 declared front cells were broken** while every gate
        was green. Real inference traffic is dynamic-N by design (one compile serves many N_token),
        but the static path is still compiled, still reachable from ``FusedTriMul``'s own default,
        and its code is shared with the dynamic path -- so leaving it untested lets a defect sit in
        code the tested path borrows from.

    COVERAGE PRIORITY
        Coverage effort over the (``N_token`` mode, ``D`` mode) product is ordered. Static-``N_token``
        uses are RARE and dynamic-``D`` uses are RARE, so:

        1. **dynamic ``N_token``, static ``D``** -- the production case; first-class, exercised
           everywhere. One compile serves many token counts, which is what real inference traffic is.
        2. dynamic ``N_token``, dynamic ``D`` -- secondary.
        3. static ``N_token``, static ``D`` -- rare; KEPT for the byte-identity / codegen probes that
           need a frozen extent. Baking the token extent costs ZERO extra compiled artifacts (``n ==
           k == D`` and K was already a compile key) and recovers 2-5%, so this row is a real lever
           rather than dead weight.
        4. static ``N_token``, dynamic ``D`` -- rarest; do not spend coverage on it by default.

        This is a COVERAGE-PRIORITY statement, not a support statement: the FIRST PRINCIPLE still
        requires every combination to WORK. It decides which cell a limited grid spends itself on.

    How this axis maps onto that ordering, and why rows 2 and 4 are ABSENT
        ``dynamic`` IS row 1 and ``static`` IS row 3. Rows 2 and 4 have no representation here, and
        that is a fact about the code rather than a gap in the pool: **no kernel in this repo can
        produce a runtime feature extent.** Measured at the front door rather than inferred from a
        convention -- ``distributed/dual_gated_gemm_a2a.py`` reads ``two_dloc = int(recv.shape[0])``
        on the HOST, derives ``Dloc = two_dloc // 2`` from it, and raises ``ValueError`` unless
        ``Dloc % postact tile_N == 0``. A marked-runtime feature extent leaves no int to read there,
        so the check could not run at all. Consistently, the only ``shape_mode``-consuming builder in
        the tree marks the token dim runtime and leaves the feature dim ``2*Dloc`` static
        DELIBERATELY, for that same reason.

        So a second ``d_shape_mode`` axis MUST NOT be declared today: it would declare a mode the
        kernels cannot emit, and every ``dynamic`` cell of it would be either unreachable or a lie.
        Reaching rows 2 and 4 is a KERNEL change (make the epilogue derive ``Dloc`` from a runtime
        extent), and the axis follows it -- never the other way round. Recorded here so that the
        absence of a D-mode axis reads as a decision with a named blocker instead of an oversight.

    Args:
        waived: Optional facet-name -> reason, for a module that genuinely cannot run one mode.
            Same semantics as :attr:`Axis.waived`: it keeps the facet visible in the report while
            exempting it, and an empty reason is rejected. Prefer an :class:`Unsupported` region
            over a waiver whenever the kernel REFUSES the mode -- a region obliges a front-door
            raise, a waiver only records a belief.

    Returns:
        The ``shape_mode`` :class:`Axis`, pool ``("static", "dynamic")``.
    """
    return Axis(
        name="shape_mode",
        domain=(
            "the mode of the TOKEN extent, and of that extent only. 'static' = it is baked into the "
            "compiled kernel (one compile per shape); 'dynamic' = it is marked runtime "
            "(mark_layout_dynamic / mark_compact_shape_dynamic), so ONE compile serves many token "
            "counts. The FEATURE extent D is STATIC under BOTH values -- that is a property of the "
            "kernels, not of this pool (see the docstring's PRIORITY section) -- so this axis is "
            "the dynamic-N_token half of the (N_token, D) mode product and never the D half. Both "
            "values are supported code paths unless the kernel declares an Unsupported region"
        ),
        values=SHAPE_MODES,
        facets={
            "static_extent": lambda m: m == "static",
            "dynamic_extent": lambda m: m == "dynamic",
        },
        waived=dict(waived or {}),
    )


def _id(value: Any) -> str:
    """Render one value as a pytest id fragment.

    Args:
        value: Any pool value. ``torch.dtype`` and similar objects stringify with a module prefix,
            which makes node ids noisy, so the leading ``torch.`` is stripped.

    Returns:
        A short string safe for a pytest node id (no spaces).
    """
    return str(value).replace("torch.", "").replace(" ", "")


@dataclass(frozen=True)
class Unsupported:
    """A region of the declared space the kernel must **RAISE** on, not compute.

    **Why raising is the property, not failing.** The hazard is a kernel that quietly accepts a
    combo it cannot compute and returns a wrong answer. Marking such a case ``xfail`` does not
    catch that: xfail only knows the test failed, so a silent wrong answer satisfies it and the
    suite stays green -- the failure mode is *rewarded*. Asserting the call raises is the only
    formulation that distinguishes "refused" from "answered wrongly". Measured: an xfail'd
    correctness assertion passes for both a raising kernel and a lying one; ``pytest.raises`` fails
    the liar with "DID NOT RAISE".

    **The raise must come from the KERNEL'S OWN API, not from wherever the failure lands.** That is
    the entire point of declaring a region: it obliges the developer to add a front-door check that
    says, in a sentence the caller can act on, that this combo is not supported. Letting a DSL ICE
    or a CUDA fault stand in for that is the failure CLAUDE.md's front-door rule already names --
    "never an ``AttributeError``/``NameError`` three frames into a dispatch". So ``raises`` is an
    exception **type**, and it must be one of :data:`API_LEVEL_ERRORS`; a region cannot be satisfied
    by an internal explosion.

    **The type check is NECESSARY BUT NOT SUFFICIENT, and the gap is measured, not theoretical.**
    ``TypeError`` is in :data:`API_LEVEL_ERRORS` and the CuTe DSL raises ``TypeError`` from inside
    ``cute.compile`` for an operand dtype it has no atom for -- so a region can be satisfied by the
    exact deep explosion it exists to forbid. See ``match`` below for the instance. The region test
    must therefore assert with :func:`front_door_raises`, which additionally requires that no DSL
    frame is on the traceback; ``pytest.raises`` alone cannot tell the two apart.

    Attributes:
        where: Predicate over **named** axis values, e.g. ``lambda N, input_dtype: ...``. Its
            parameter names must be declared axes -- they are introspected to decide which axes the
            region needs, so a typo raises rather than silently matching nothing. Must be total
            over the pool; an exception inside it fails at import.
        raises: The exception **type** the kernel's own validation raises. Must be in
            :data:`API_LEVEL_ERRORS`. Naming a toolchain exception is rejected, with a message
            saying to add the guard instead -- which is the work the region exists to force.
        match: Regex the message must match, passed to ``pytest.raises(match=)``. Make it name the
            constraint, not just the symptom: a pattern loose enough to match an unrelated failure
            proves nothing. Required.

            **Measured instance of that failing.** Before `GemmLayerNormGemmSm90` gated its
            activation dtype, an fp32 activation produced ``TypeError: unsupported a_dtype and
            b_dtype, got Float32 and Float32`` -- raised from
            ``cutlass/utils/hopper_helpers.py:167`` inside ``cute.compile``, thirteen frames deep.
            ``TypeError`` is in :data:`API_LEVEL_ERRORS`, so a region written
            ``raises=TypeError, match=r"unsupported.*dtype"`` passed with NO front-door check at
            all: the toolchain's own explosion satisfied it, which is the state the region existed
            to prevent. Only a hand-chosen ``match`` kept it out, and nothing obliged the ``match``
            to be well chosen. :func:`front_door_raises` is what obliges it now -- and it is what
            the region test must use, precisely because a ``match`` cannot be trusted to carry this
            on its own.
        reason: Why the combo is unsupported. Required prose; this is the part no error message can
            reconstruct.
    """

    where: Callable[..., bool]
    raises: type
    match: str
    reason: str

    def __post_init__(self) -> None:
        """Reject a region that cannot be checked or explained.

        Returns:
            None.

        Raises:
            ValueError: If ``raises`` or ``reason`` is empty, or ``where`` takes no named axes.
        """
        import inspect

        if not (isinstance(self.raises, type) and issubclass(self.raises, BaseException)):
            raise ValueError(f"Unsupported.raises must be an exception TYPE, got {self.raises!r}")
        if self.raises not in API_LEVEL_ERRORS:
            raise ValueError(
                f"Unsupported.raises={self.raises.__name__} is not one of "
                f"{[e.__name__ for e in API_LEVEL_ERRORS]}. A region must be satisfied by the "
                f"KERNEL refusing at its own front door -- not by a DSL ICE, a CUDA fault, or an "
                f"AttributeError three frames into a dispatch. If the kernel currently explodes "
                f"internally on this combo, that IS the bug: add a check to the public entry that "
                f"raises one of the above with a sentence the caller can act on."
            )
        if not (self.match and self.match.strip()):
            raise ValueError("Unsupported.match must be a non-empty regex naming the constraint")
        if not (self.reason and self.reason.strip()):
            raise ValueError("Unsupported.reason must say why the combo is unsupported")
        if not inspect.signature(self.where).parameters:
            raise ValueError("Unsupported.where must take named axis parameters, e.g. (N, dtype)")

    def axis_names(self) -> Tuple[str, ...]:
        """The axes this region's predicate reads.

        Returns:
            The parameter names of ``where``, in declaration order. Used to decide whether a given
            parametrization has enough axes to evaluate the region at all.
        """
        import inspect

        return tuple(inspect.signature(self.where).parameters)


@dataclass(frozen=True)
class _NoUnsupported:
    """Sentinel: this kernel has no unsupported combos, and here is why. See :func:`no_unsupported`.

    Attributes:
        because: The reasoning. Required -- the whole point is that "there are none" must be a
            claim someone made, not a field nobody filled in.
    """

    because: str


def no_unsupported(*, because: str) -> _NoUnsupported:
    """Declare that every combo in the declared pools is supported, with the reasoning.

    ``unsupported=`` is mandatory on :class:`KernelMatrix`. This is how a kernel that genuinely has
    no unsupported region satisfies it, so that "none" and "nobody thought about it" are different
    states in the source.

    Args:
        because: Why there are none. Must be non-empty. Say what makes the whole declared space
            computable -- not "no known issues", which is the absence of thought restated.

    Returns:
        A sentinel accepted by ``KernelMatrix(unsupported=...)``.

    Raises:
        ValueError: If ``because`` is empty or whitespace.
    """
    if not (because and because.strip()):
        raise ValueError("no_unsupported() requires because='...': state why nothing is excluded")
    return _NoUnsupported(because)


#: What a kernel COMPUTES, in the only terms that predict which input distribution can hide a bug in
#: it, mapped to the axis every such kernel must sample.
#:
#: **This table is the answer to "which shapes did nobody think of", one level up.** ``torch.randn``
#: is zero-mean, which is the most benign point of a LayerNorm's ``|mu|/sigma`` conditioning -- and
#: this repo has already paid for that: the padded-tile variance bug was invisible at the zero mean
#: ``torch.randn`` produces and survived 598 tests. A pool can be wide in every SHAPE and still be
#: one point wide in the property that matters.
#:
#: Keyed by the computation rather than by the kernel name, deliberately. A name-keyed table has to
#: be edited for every new kernel and silently requires nothing when somebody forgets; a
#: trait-keyed one is reached through ``KernelMatrix(computes=...)``, which is mandatory, so a new
#: kernel cannot arrive without answering the question.
#: **Only a trait whose property has been MEASURED to matter maps to a non-empty tuple.** The other
#: traits exist and are declared by real matrices, but require nothing yet -- turning one on is a
#: one-line edit here that immediately obliges every matrix declaring it, which is the point of
#: keying on the trait rather than on the kernel. Requiring a property nobody has measured would
#: produce a wave of waivers written to make an error go away, and a waiver with no evidence behind
#: it is worse than no requirement: it reads as a decision.
SENSITIVITY: Dict[str, Tuple[str, ...]] = {
    #: A mean/variance reduction over a row. MEASURED: standalone ``layernorm_fwd``'s error grows
    #: ~850x from ``mu=0`` to ``mu=100``, monotonically. This is the one that has already cost this
    #: repo a shipped bug, so it is the one that is enforced.
    "row_reduction": ("row_mean",),
    #: A saturating activation (sigmoid / GLU). Far into a tail the derivative vanishes, so a wrong
    #: pre-activation can produce an output indistinguishable from the right one.
    #: NOT YET REQUIRED -- it will imply ``activation_scale`` once that has been swept. Note the
    #: existing scalings are not it: ``* 0.1`` in `test_dual_gated_gemm.py` puts the pre-activation
    #: in the sigmoid's RESPONSIVE range, i.e. it samples the one regime where the gate is easiest.
    "saturating_activation": (),
    #: fp8 operands. Four exponent bits, so a scale that is unremarkable at bf16 flushes to zero or
    #: saturates -- and both look like a correct kernel fed a bad input.
    #: NOT YET REQUIRED -- it will imply ``dynamic_range``. The ``/ 4`` in the GEMM tests is an
    #: fp8-range device, not a sweep of it.
    "fp8_operands": (),
    #: A plain contraction with no reduction over a normalized row and no saturation. Nothing here
    #: is distribution-sensitive beyond magnitude, which the shape axes already vary.
    "contraction": (),
    #: Pure data movement -- a transpose, a staged copy, a permutation. The correct result IS the
    #: input, so it is bitwise-checkable and no distribution can hide anything.
    "data_movement": (),
}


@dataclass(frozen=True)
class _ComputesNothingNumeric:
    """Sentinel: this matrix governs no numeric computation, and here is why.

    Attributes:
        because: The reasoning. Required, for the same reason :class:`_NoUnsupported`'s is: "there
            is nothing to sample" must be a claim somebody made rather than a field left blank.
    """

    because: str


def computes_nothing_numeric(*, because: str) -> _ComputesNothingNumeric:
    """Declare that a matrix's subject computes no distribution-sensitive value, with the reasoning.

    ``computes=`` is mandatory on :class:`KernelMatrix`. This is how a matrix whose subject is a
    dispatcher decision, a layout, or a cache key satisfies it, so that "nothing to sample" and
    "nobody thought about it" are different states in the source.

    Args:
        because: Why no input property applies. Must be non-empty. Say what the subject actually
            produces -- "returns a config, launches nothing" -- not "not applicable".

    Returns:
        A sentinel accepted by ``KernelMatrix(computes=...)``.

    Raises:
        ValueError: If ``because`` is empty or whitespace.
    """
    if not (because and because.strip()):
        raise ValueError(
            "computes_nothing_numeric() requires because='...': state what the subject produces"
        )
    return _ComputesNothingNumeric(because)


#: Exception types that count as a kernel refusing **at its own front door**. Deliberately narrow:
#: these are what a Python API raises on purpose, after checking its arguments. Anything else --
#: a DSL ICE, a CUDA launch failure, an ``AttributeError`` three frames into a dispatch -- is the
#: kernel *failing*, not *refusing*, and is exactly what CLAUDE.md's front-door rule forbids being
#: passed off as a rejection. A region that names one of those is telling you to go add a guard.
#:
#: **AssertionError is deliberately NOT here.** ``assert`` is stripped by ``python -O``, so a guard
#: written that way vanishes under optimization and the internal explosion comes back -- a region
#: satisfied by one is guaranteeing nothing. Write ``raise ValueError(...)``. (Plain argument
#: asserts elsewhere in the repo are fine; this is about the combos a region promises are refused.)
API_LEVEL_ERRORS = (ValueError, TypeError, NotImplementedError)

#: Path COMPONENTS that mean "this frame is the CuTe DSL". Matched as whole path parts and never as
#: a substring, so a source tree that happens to contain the letters is not mistaken for the
#: toolchain. The measured frame this was built against is
#: ``.../site-packages/nvidia_cutlass_dsl/python_packages/cutlass/utils/hopper_helpers.py``, whose
#: parts include both names.
_DSL_PATH_PARTS = frozenset({"cutlass", "nvidia_cutlass_dsl"})

#: The importable package's root directory. ``kernel_matrix.py`` lives at
#: ``fold_cp_ops/testing/kernel_matrix.py``, so two parents up is ``fold_cp_ops/``. Derived rather
#: than written down, for the same reason ``cache_utils`` derives its fingerprint root: a moved file
#: must not silently change what counts as "our code".
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _is_under(filename: str, root: pathlib.Path) -> bool:
    """Whether a traceback filename lives under ``root``.

    Args:
        filename: A frame's ``__file__``-style path, as ``traceback.extract_tb`` reports it. A
            synthetic frame (``"<string>"``, ``"<stdin>"``) resolves to a path under the CWD, which
            is not under the package root -- so exec'd code reads as third-party, which is right.
        root: The directory to test against. Resolved by the caller.

    Returns:
        True if ``filename`` is ``root`` or sits inside it.
    """
    try:
        p = pathlib.Path(filename).resolve()
    except (OSError, ValueError):  # pragma: no cover - a path the OS refuses to resolve
        return False
    return p == root or root in p.parents


def front_door_problem(filenames: Sequence[str]) -> str | None:
    """Judge whether a raise came from a kernel's own front door, from its traceback's file paths.

    Purpose
        `API_LEVEL_ERRORS` makes a region's exception TYPE checkable. It cannot make the region's
        real claim checkable, which is that the kernel *refused at its entry point*. This is the
        missing half.

    Semantics
        Two rules, applied in this order, on the filenames of one traceback in call order:

        1. **No frame may belong to the CuTe DSL.** A DSL frame anywhere on the path means the
           kernel was already being COMPILED when it complained -- the check is latent, not a front
           door, and it fires only for a caller who got far enough to trigger a compile. This rule
           is the one that matters, because it catches the case rule 2 cannot: our OWN
           ``@cute.jit`` body raising during tracing has a raising frame under the package root and
           is still a latent check.
        2. **The raising frame must be under the package root.** Catches a refusal that is really a
           third-party library's -- a ``ValueError`` out of torch, or a test helper failing before
           it ever reached the kernel.

        **"Our code" is the whole package, not the entry module, and that is deliberate.** A front
        door legitimately delegates: `gemm()` refuses a bad dtype through
        ``tensor_contract.check_tensor`` and a bad stride through
        ``gemm_tvm_ffi_utils._raise_misaligned``, both of which are shared helpers written to be
        called from several entries. Requiring the literal ``raise`` in the entry module would
        force every check to be inlined, which is worse code for no gain -- what makes those
        helpers front-door checks is that nothing has been traced yet, and rule 1 is what tests
        that. So the answer to "is a helper still the front door?" is YES, provided the path to it
        never entered the DSL.

        **A check that genuinely cannot run before tracing is not a front-door check.** If some
        future entry must compile something before it can validate, this will reject it -- and that
        rejection is correct: the region should then be re-examined rather than the rule relaxed,
        because a caller only reaches such a check by paying the compile.

    Args:
        filenames: One traceback's frame filenames, oldest first, as
            ``[f.filename for f in traceback.extract_tb(tb)]``. Must be non-empty; an empty
            sequence means the caller has no traceback to judge and is refused rather than passed.

    Returns:
        None when the raise is a front-door raise, or a human-readable problem string naming what
        was found. The string is the failure message a caller should surface verbatim.

    Warning:
        **This checks PROVENANCE ONLY, and provenance is necessary but not sufficient.** A caller
        that wants the full guarantee must ALSO check the exception's TYPE against
        :data:`API_LEVEL_ERRORS`. The two are independent: a bare ``KeyError`` raised from our own
        frame with no DSL frame on the traceback returns None here and is still unusable, because
        it names a torch dtype and no argument. Measured -- a latent-raise sweep classified
        ``KeyError: torch.float64`` as a front-door raise on its first pass using this function
        alone, and caught it only because a ``KeyError`` looked wrong in the output.

        :func:`front_door_raises` is safe because ``pytest.raises`` supplies the type half
        separately; anything calling this predicate directly does not get that.

        **The converse gap is why the type check alone is also insufficient**, and it has a live
        instance: the CuTe DSL raises ``TypeError`` from inside ``cute.compile`` for an operand
        dtype it has no atom for, and ``TypeError`` IS in :data:`API_LEVEL_ERRORS`. So a region
        written ``raises=TypeError, match=r"unsupported.*dtype"`` passes against that deep
        explosion with no front-door check at all. Provenance is what rejects it. Neither half
        substitutes for the other, which is the whole reason both exist.

    Raises:
        ValueError: If ``filenames`` is empty.
    """
    if not filenames:
        raise ValueError(
            "front_door_problem() needs at least one traceback frame; an empty traceback cannot be "
            "judged and passing it would report every raise as a front-door raise."
        )
    for name in filenames:
        parts = set(pathlib.Path(name).parts)
        if parts & _DSL_PATH_PARTS:
            return (
                f"the exception was raised from inside the CuTe DSL -- frame {name!r} is on the "
                f"traceback, so the kernel was already COMPILING when it complained. That is a "
                f"LATENT check, not a front-door check: it fires only for a caller who got far "
                f"enough to trigger a compile, it costs that compile before saying no, and its "
                f"message is the toolchain's rather than one naming the argument to change. Note "
                f"the exception TYPE was acceptable -- membership in "
                f"{[e.__name__ for e in API_LEVEL_ERRORS]} is NECESSARY BUT NOT SUFFICIENT, which "
                f"is exactly why this check exists. Add an explicit check at the kernel's own "
                f"entry point that raises before anything is traced."
            )
    last = filenames[-1]
    if not _is_under(last, _PACKAGE_ROOT):
        return (
            f"the exception was raised in {last!r}, which is not under {str(_PACKAGE_ROOT)!r}. A "
            f"declared unsupported region is a promise that THIS package refuses the combination; "
            f"a refusal that comes out of a third-party library (or out of the test's own helper "
            f"before it reached the kernel) is not that promise, even when the type happens to be "
            f"one of {[e.__name__ for e in API_LEVEL_ERRORS]}. Add an explicit check at the "
            f"kernel's own entry point."
        )
    return None


@contextlib.contextmanager
def front_door_raises(expected_error, match):
    """Assert a block raises ``expected_error`` matching ``match`` **from the kernel's front door**.

    Purpose
        A drop-in replacement for ``pytest.raises(expected_error, match=match)`` in an
        unsupported-region test, adding the one property the type check cannot express: that the
        author wrote an explicit check at the entry point instead of leaving the caller to a latent
        explosion deeper in.

        **Measured, on this repo.** Before `GemmLayerNormGemmSm90` gated its activation dtype, an
        fp32 activation produced::

            TypeError: unsupported a_dtype and b_dtype, got Float32 and Float32

        raised from ``cutlass/utils/hopper_helpers.py`` inside ``cute.compile``. ``TypeError`` is in
        `API_LEVEL_ERRORS`, so a region written ``raises=TypeError, match=r"unsupported.*dtype"``
        passed with NO front-door check at all -- satisfied by the exact deep explosion the region
        exists to forbid. Only a carefully-chosen ``match`` kept it out, and nothing enforced that
        the ``match`` be carefully chosen. This does.

    Semantics
        Runs ``pytest.raises`` first, so an un-raising kernel still fails with "DID NOT RAISE" and
        a wrong type or message still fails the way it always has. Only once those pass is the
        traceback's provenance judged, by `front_door_problem`; a problem there becomes an
        ``AssertionError`` naming the frame and what to do about it.

        The ordering matters: provenance is the LAST thing checked, so a failure message always
        describes the most specific thing that is wrong rather than complaining about where an
        exception of the wrong type came from.

    Args:
        expected_error: The exception type the kernel's own guard must raise. Should be one of
            `API_LEVEL_ERRORS` -- `Unsupported` already refuses anything else at declaration time,
            and this does not re-check it, because a caller using this outside a region is entitled
            to assert on whatever type their front door raises.
        match: Regex the message must match, passed straight to ``pytest.raises(match=)``. Still
            required and still worth writing well: this makes a lazy ``match`` survivable rather
            than making it unnecessary. Name the constraint, not the symptom.

    Yields:
        The ``ExceptionInfo`` ``pytest.raises`` yields, so ``as excinfo`` keeps working.

    Raises:
        AssertionError: If nothing was raised, if the type or message did not match (both from
            ``pytest.raises``), or if the raise did not come from the front door.

    Note:
        The name ending in ``raises`` is load-bearing, not decorative:
        `audit_test_module`'s `_has_working_unsupported_test` recognises a real region test by
        finding a call whose function name ends in ``raises`` and whose first argument is
        ``expected_error``. Renaming this to something that does not end in ``raises`` would make
        every module using it look like it had no region test at all.
    """
    with pytest.raises(expected_error, match=match) as excinfo:
        yield excinfo
    problem = front_door_problem([f.filename for f in _traceback.extract_tb(excinfo.tb)])
    if problem:
        raise AssertionError(
            f"{expected_error.__name__} was raised and matched {match!r}, but not from a front-door "
            f"check: {problem}"
        )


#: Marker for "the caller did not pass ``unsupported=`` at all", distinct from passing an empty
#: tuple. Omission is the failure mode this whole field exists to prevent, so it must be detectable.
_UNSET = object()


@dataclass(frozen=True)
class KernelMatrix:
    """The declared test matrix for one kernel: the only sanctioned source of test values.

    Attributes:
        kernel: Kernel name, e.g. ``"layernorm"``. Used as the registry key and in messages.
        axes: The declared axes. Order is irrelevant; lookup is by name.
        unsupported: **Required.** Either a non-empty tuple of :class:`Unsupported` regions, or
            ``no_unsupported(because=...)``. Omitting it raises. A kernel that never states which
            combos it refuses is a kernel that can silently accept one and return a wrong answer;
            making the field mandatory converts "nobody knew" into "nobody wrote it down", which is
            a much smaller gap and one a reviewer can see.
        computes: **Required.** Either a non-empty tuple of trait names from :data:`SENSITIVITY`, or
            ``computes_nothing_numeric(because=...)``. Each trait implies the input properties this
            kernel must SAMPLE, and every implied property must be a declared axis or carry a
            waiver in ``property_waivers``. Shape coverage is not distribution coverage: a pool can
            be wide in every extent and still be one point wide in the row mean, which is exactly
            how the padded-tile variance bug survived 598 tests.
        property_waivers: Reason strings, keyed by property name, for the implied properties this
            kernel does NOT sample. A waiver is a MEASURABLE claim -- "the chain is insensitive" --
            so the reason should name the evidence, e.g. the test that swept it. Waiving without
            measuring is how a property becomes decoration.
    """

    kernel: str
    axes: Tuple[Axis, ...]
    unsupported: Any = _UNSET
    computes: Any = _UNSET
    property_waivers: Mapping[str, str] = field(default_factory=dict)

    #: Why this kernel declares no argument-fault axis, or None if it must. A kernel whose tests
    #: drive a functor directly -- no front door, no optional tensors, ``epilogue_args=()`` -- has
    #: no argument to malform, and saying so is different from forgetting to check.
    arg_faults_waived: str | None = None

    def __post_init__(self) -> None:
        """Validate the declaration and register the matrix.

        Returns:
            None.

        Raises:
            ValueError: If two axes share a name; if a second matrix claims the same kernel name;
                if ``unsupported`` was omitted or is an empty tuple; if a region names an axis that
                does not exist; or if a region matches nothing in the pool, which would make it
                untestable and therefore decoration.
        """
        import itertools

        names = [a.name for a in self.axes]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.kernel}: duplicate axis name in {names}")

        if self.unsupported is _UNSET or self.unsupported == ():
            raise ValueError(
                f"{self.kernel}: `unsupported=` is REQUIRED. Declare the combos the kernel must "
                f"RAISE on, e.g. Unsupported(where=lambda N, input_dtype: ..., raises=..., "
                f"reason=...), or state that there are none with "
                f"unsupported=no_unsupported(because='...'). Leaving it unset is how a kernel "
                f"silently accepts a combo it cannot compute and returns a wrong answer."
            )
        for region in self.regions():
            missing = [n for n in region.axis_names() if n not in names]
            if missing:
                raise ValueError(
                    f"{self.kernel}: unsupported region reads undeclared axes {missing}; "
                    f"declared: {names}"
                )
            pools = [self.axis(n).values for n in region.axis_names()]
            if not any(
                region.where(**dict(zip(region.axis_names(), c))) for c in itertools.product(*pools)
            ):
                raise ValueError(
                    f"{self.kernel}: unsupported region {region.reason!r} matches NO combo in the "
                    f"declared pools, so nothing can test it. Add values that reach it, or drop "
                    f"the region."
                )

        if self.computes is _UNSET or self.computes == ():
            raise ValueError(
                f"{self.kernel}: `computes=` is REQUIRED. Name what this kernel computes, from "
                f"{sorted(SENSITIVITY)}, e.g. computes=('row_reduction', 'contraction') -- each "
                f"trait implies an input property the pool must SAMPLE. A matrix whose subject "
                f"computes nothing numeric says so with computes_nothing_numeric(because='...'). "
                f"Shape coverage is not distribution coverage: torch.randn is zero-mean, which is "
                f"the one row mean at which a padded-tail variance defect contributes exactly zero."
            )
        if not isinstance(self.computes, _ComputesNothingNumeric):
            unknown = [t for t in self.computes if t not in SENSITIVITY]
            if unknown:
                raise ValueError(
                    f"{self.kernel}: unknown computation trait(s) {unknown}; declared traits are "
                    f"{sorted(SENSITIVITY)}. Add the trait to SENSITIVITY with the property it "
                    f"implies rather than inventing a name here, or the requirement is silent."
                )
            for prop in self.required_properties():
                if prop in names:
                    continue
                waiver = self.property_waivers.get(prop, "")
                if not waiver.strip():
                    raise ValueError(
                        f"{self.kernel}: computes {list(self.computes)}, which requires sampling "
                        f"{prop!r}, but there is no axis by that name and no entry in "
                        f"property_waivers. Declare an Axis({prop!r}, ...) with facets for the "
                        f"regimes, or waive it with MEASURED evidence: "
                        f"property_waivers={{{prop!r}: 'measured insensitive: <how>'}}."
                    )
        stray = [p for p in self.property_waivers if p not in self.required_properties()]
        if stray:
            raise ValueError(
                f"{self.kernel}: property_waivers names {stray}, which nothing requires. A waiver "
                f"for a property that was never required reads as coverage that is not there."
            )

        if self.kernel in REGISTRY and REGISTRY[self.kernel] is not self:
            raise ValueError(f"a different matrix is already registered for {self.kernel!r}")
        REGISTRY[self.kernel] = self

    def required_properties(self) -> Tuple[str, ...]:
        """The input properties this kernel's declared traits oblige it to sample.

        Returns:
            Property names, de-duplicated and in a stable order. Empty when the matrix declared
            ``computes_nothing_numeric(...)`` or only traits that imply nothing -- which is a
            claim someone made, never an unfilled field, because ``computes=`` is mandatory.
        """
        if isinstance(self.computes, _ComputesNothingNumeric):
            return ()
        seen: Dict[str, None] = {}
        for trait in self.computes:
            for prop in SENSITIVITY.get(trait, ()):
                seen[prop] = None
        return tuple(seen)

    def regions(self) -> Tuple[Unsupported, ...]:
        """The declared unsupported regions.

        Returns:
            The regions, or an empty tuple when the kernel declared ``no_unsupported(...)``. Empty
            here means "declared none with a reason", never "field not filled in" -- that case is
            rejected at construction.
        """
        return () if isinstance(self.unsupported, _NoUnsupported) else tuple(self.unsupported)

    def axis(self, name: str) -> Axis:
        """Look up one axis by name.

        Args:
            name: Axis name.

        Returns:
            The :class:`Axis`.

        Raises:
            KeyError: If undeclared, listing what is available -- almost always a typo in a
                ``parametrize`` call, and worth catching at import rather than as an empty grid.
        """
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError(f"{self.kernel}: no axis {name!r}; declared: {[a.name for a in self.axes]}")

    def _select(self, name: str, only, drop) -> Tuple[Any, ...]:
        """Resolve one axis's values for a single test, honouring ``only``/``drop``.

        Args:
            name: Axis name.
            only: Mapping of axis name -> the exact values to keep, or None.
            drop: Mapping of axis name -> values to remove, or None.

        Returns:
            The selected values, in pool order so ids stay stable regardless of how the restriction
            was written.

        Raises:
            ValueError: If a requested value is not in the pool, or the selection is empty. Values
                outside the pool are refused rather than added, because a test that can invent
                shapes is a test the diversity check never sees -- add it to the matrix instead.
        """
        pool = self.axis(name).values
        chosen = pool
        if only and name in only:
            wanted = tuple(only[name])
            outside = [v for v in wanted if not any(v is p or v == p for p in pool)]
            if outside:
                raise ValueError(
                    f"{self.kernel}.{name}: {outside} are not in the declared pool. Add them to "
                    f"the matrix (so the diversity check sees them) rather than to one test."
                )
            chosen = tuple(p for p in pool if any(p is w or p == w for w in wanted))
        if drop and name in drop:
            unwanted = tuple(drop[name])
            chosen = tuple(p for p in chosen if not any(p is u or p == u for u in unwanted))
        if not chosen:
            raise ValueError(f"{self.kernel}.{name}: selection is empty")
        return chosen

    def parametrize(
        self,
        *names: str,
        only: Mapping[str, Sequence[Any]] | None = None,
        drop: Mapping[str, Sequence[Any]] | None = None,
        cells: Iterable[Sequence[Any]] | None = None,
        because: str | None = None,
    ):
        """Build the ``pytest.mark.parametrize`` for a test from this matrix.

        Args:
            *names: Axis names to parametrize over, in the order the ids should read. Each must be
                declared and must match the test function's parameter name.
            only: Axis name -> exact values to keep. Requires ``because``.
            drop: Axis name -> values to remove. Requires ``because``.
            cells: An explicit list of value tuples, one per ``names`` entry, used instead of the
                cross product. For grids that are a *list of cells* rather than a product -- a perf
                gate pins one number per cell, so a product would be hundreds of timed runs. Every
                component is still checked against its pool. Requires ``because``, and is mutually
                exclusive with ``only``/``drop``.
            because: Why this test departs from the full matrix. **Required whenever it does**, and
                must be a non-empty string. This is the whole point: a restriction should cost one
                sentence, so that forgetting a config and excluding it look different.

        Returns:
            A decorator applying a single ``pytest.mark.parametrize`` over the selected grid.

        Raises:
            ValueError: If a restriction is given without ``because``; if ``because`` is given
                without a restriction (it would be describing nothing); if ``cells`` is combined
                with ``only``/``drop``; if a cell has the wrong arity or a value outside its pool.
            KeyError: If an axis name is undeclared.
        """
        restricted = bool(only or drop or cells is not None)
        if restricted and not (because and because.strip()):
            raise ValueError(
                f"{self.kernel}.parametrize{names}: restricting the matrix requires because='...' "
                f"-- say why these configs are excluded so a gap and a decision look different."
            )
        if because and not restricted:
            raise ValueError(
                f"{self.kernel}.parametrize{names}: because='{because}' but nothing is restricted."
            )
        if cells is not None and (only or drop):
            raise ValueError(f"{self.kernel}.parametrize{names}: cells= excludes only=/drop=")

        if cells is not None:
            grid = [tuple(c) for c in cells]
            for c in grid:
                if len(c) != len(names):
                    raise ValueError(
                        f"{self.kernel}: cell {c} has arity {len(c)}, expected {names}"
                    )
                for axis_name, value in zip(names, c):
                    pool = self.axis(axis_name).values
                    if not any(value is p or value == p for p in pool):
                        raise ValueError(
                            f"{self.kernel}.{axis_name}: {value!r} in cell {c} is not in the "
                            f"declared pool. Add it to the matrix, not to one test."
                        )
        else:
            grid: list[tuple] = [()]
            for name in names:
                grid = [row + (v,) for row in grid for v in self._select(name, only, drop)]

        # A supported-path test must not assert correctness on a combo declared UNSUPPORTED --
        # those two claims contradict each other, and the contradiction is invisible otherwise.
        # Regions needing an axis this test does not parametrize cannot be evaluated, so are skipped.
        for row in grid:
            bound = dict(zip(names, row))
            for region in self.regions():
                if set(region.axis_names()) <= set(names) and region.where(
                    **{k: bound[k] for k in region.axis_names()}
                ):
                    raise ValueError(
                        f"{self.kernel}.parametrize{names}: the cell {bound} lies in the "
                        f"unsupported region {region.reason!r}, which must RAISE. A correctness "
                        f"test cannot cover it -- exclude it here, and cover it with "
                        f"parametrize_unsupported()."
                    )

        _record_emission(self.kernel, names, grid)
        ids = ["-".join(f"{n}{_id(v)}" for n, v in zip(names, row)) for row in grid]
        values = [row[0] if len(names) == 1 else row for row in grid]
        return pytest.mark.parametrize(",".join(names), values, ids=ids)

    def parametrize_unsupported(self, *names: str):
        """Parametrize over exactly the pool combos that must RAISE, with the expected pattern.

        Emits one cell per combo of ``names`` that lands in a declared :class:`Unsupported` region,
        appending ``expected_error`` (the exception type the kernel's own guard must raise) and
        ``expected_match`` (the regex its message must match).

        **A cell may land in SEVERAL regions, and then the expectation admits ALL of them** --
        ``expected_error`` is a tuple of types and ``expected_match`` an alternation. ``pytest.raises``
        accepts a tuple, so :func:`front_door_raises` needs no special case.

        Why that is the correct semantics and not a weakening: **a region's claim is "this
        configuration must be REFUSED at the front door". Which guard fires is the kernel's internal
        check order, which the matrix does not know and has no business pinning.** A cell violating
        two constraints must still be refused; asserting *which* refusal appears is asserting an
        implementation detail that is free to change. The claim given up -- "refused for THIS
        reason" -- is only ever meaningful for a cell in exactly one region, and those are untouched
        (in the matrix that motivated this, 248 of 284).

        **This replaces a first-match ``break``, which was a defect and not a simplification.** The
        expectation was decided by REGION DECLARATION order in the matrix while the raise was decided
        by GUARD order in the kernel, and nothing reconciled them. Measured cost when the two
        disagreed: ``pytest.raises`` did not catch the exception, so it propagated *inside a test body
        holding a collective* -- the raising ranks left, the rest blocked, and a 16-rank job desynced
        at teardown. 19 failures and one hang, one defect. The overlap that caused it was statically
        present from the start (36 of 284 cells) and is now recorded via :func:`unsupported_overlap`,
        with the all-regions-admitted invariant asserted at the emission site so a revert to
        first-match cannot be silent.

        **The test body calls the kernel inside**
        ``front_door_raises(expected_error, expected_match)``. Use :func:`front_door_raises`, not
        ``pytest.raises``: the two agree on the type and the message, and only the former also
        rejects a raise that came from inside ``cute.compile``. That difference is the whole
        enforcement -- ``TypeError`` is an API-level type AND the type the DSL raises for an
        unsupported operand dtype, so a region asserted with bare ``pytest.raises`` can pass
        against a kernel with no front-door check at all. Measured; see :class:`Unsupported`.

        Args:
            *names: Axis names to sweep. Must be a superset of the axes every declared region
                reads, or the regions it cannot evaluate would be silently skipped and the test
                would claim more coverage than it has.

        Returns:
            A decorator applying ``pytest.mark.parametrize`` over ``(*names, "expected_error")``.

        Raises:
            KeyError: If an axis name is undeclared.
            ValueError: If the kernel declared ``no_unsupported(...)`` -- there is nothing to
                sweep, and a test asserting otherwise would never run; if ``names`` cannot evaluate
                some region; or if the sweep is empty, which means the pool cannot reach the region.
        """
        import itertools

        regions = self.regions()
        if not regions:
            raise ValueError(
                f"{self.kernel}: declared no_unsupported(because={self.unsupported.because!r}), so "
                f"there is nothing for parametrize_unsupported() to sweep. Remove this test, or "
                f"declare the region it was written for."
            )
        blind = [r for r in regions if not set(r.axis_names()) <= set(names)]
        if blind:
            raise ValueError(
                f"{self.kernel}.parametrize_unsupported{names} cannot evaluate region(s) "
                f"{[r.reason for r in blind]}, which need axes "
                f"{sorted({a for r in blind for a in r.axis_names()})}. Sweep those axes too, or "
                f"the test silently covers less than it appears to."
            )

        grid = []
        overlapping = []
        for row in itertools.product(*(self.axis(n).values for n in names)):
            bound = dict(zip(names, row))
            hits = [r for r in regions if r.where(**{k: bound[k] for k in r.axis_names()})]
            if not hits:
                continue
            if len(hits) > 1:
                overlapping.append((row, tuple(r.raises.__name__ for r in hits)))
            # EVERY matching region, not the first. See the method docstring for why the old
            # `break` was a defect rather than a simplification.
            raises = tuple(dict.fromkeys(r.raises for r in hits))
            expected_error = raises[0] if len(raises) == 1 else raises
            expected_match = (
                hits[0].match if len(hits) == 1 else "|".join(f"(?:{r.match})" for r in hits)
            )
            grid.append(row + (expected_error, expected_match))
            # The invariant, asserted where it is established rather than left to a reader: the
            # emitted expectation must admit EVERY region this cell matches. Re-introducing the
            # first-match `break` trips this immediately, at import, on the matrix that has overlap
            # -- which is the whole point of checking the invariant instead of checking for overlap.
            # A pure overlap check would have to be silenced once overlap became legal, and a
            # silenced check is how this defect survived: 36 overlapping cells sat in one matrix,
            # statically visible, for as long as the machinery existed.
            admitted = expected_error if isinstance(expected_error, tuple) else (expected_error,)
            missing = [r.raises.__name__ for r in hits if r.raises not in admitted]
            if missing:
                raise AssertionError(
                    f"{self.kernel}.parametrize_unsupported{names}: cell {bound} matches "
                    f"{len(hits)} regions but the emitted expectation admits only "
                    f"{[t.__name__ for t in admitted]}, dropping {missing}. The kernel refuses for "
                    f"whichever guard it checks FIRST, which the matrix does not know and must not "
                    f"encode -- so the expectation has to admit every matching region. Collect all "
                    f"matches instead of breaking on the first."
                )
        if not grid:
            raise ValueError(f"{self.kernel}.parametrize_unsupported{names}: no combo matched")
        if overlapping:
            _record_overlap(self.kernel, names, overlapping)

        _record_emission(self.kernel, names, [row[: len(names)] for row in grid])
        ids = ["-".join(f"{n}{_id(v)}" for n, v in zip(names, row)) for row in grid]
        return pytest.mark.parametrize(
            ",".join(names + ("expected_error", "expected_match")), grid, ids=ids
        )


def int_facets(*, tile: int, big: int, small: int) -> Dict[str, Callable[[int], bool]]:
    """The standard diversity facets for an integer extent axis, with kernel-specific boundaries.

    Shared because the *shape* of the question is the same for every kernel -- is the pool all
    powers of two, all even, all tile-aligned, all mid-sized -- while the boundaries are not. Pass
    the kernel's own switch points rather than reusing another kernel's.

    Args:
        tile: Tile width in elements. ``tile_aligned`` tests ``v % tile == 0``, i.e. whether the
            trailing-tile predicate is engaged. Must be positive.
        big: The extent above which the kernel changes geometry (a wider block, a re-read, a
            cluster). ``above_switch`` tests ``v > big``.
        small: The extent at or below which the kernel uses its narrowest thread-per-row rung.
            ``small_rung`` tests ``v <= small``.

    Returns:
        A mapping usable as ``Axis.facets``. Five facets: ``power_of_two``, ``odd``,
        ``tile_aligned``, ``above_switch``, ``small_rung``.
    """
    return {
        "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
        "odd": lambda v: v % 2 == 1,
        "tile_aligned": lambda v: v % tile == 0,
        "above_switch": lambda v: v > big,
        "small_rung": lambda v: v <= small,
    }


def dtype_facets(dtypes: Sequence[Any]) -> Dict[str, Callable[[Any], bool]]:
    """One facet per dtype, so a pool that quietly drops one scores zero on it.

    Args:
        dtypes: The dtypes the kernel accepts. Each becomes a facet named after it, with the
            ``torch.`` prefix stripped for readability.

    Returns:
        A mapping usable as ``Axis.facets``. A missing dtype gives ``p = 0`` and therefore
        ``H = 0``, which fails the threshold and names the dtype in the message.
    """
    return {str(d).replace("torch.", ""): (lambda v, d=d: v == d) for d in dtypes}


def _has_working_unsupported_test(tree) -> bool:
    """Whether the module has a test that both sweeps the regions AND asserts the raise.

    Checked on the AST rather than by searching the source text, because a substring search is
    satisfied by a comment or a docstring mentioning the name -- measured: a module whose only
    reference was ``# TODO: add @M.parametrize_unsupported("N") one day`` passed. And the decorator
    alone is not enough either: a body that never calls the kernel inside
    ``pytest.raises(expected_error, ...)`` sweeps every refused combo and asserts nothing about
    them, which is the vacuous-guard shape this whole module exists to prevent.

    Args:
        tree: A parsed module AST.

    Returns:
        True if some ``def test_*`` carries a ``parametrize_unsupported`` decorator and its body
        contains a ``pytest.raises`` whose first argument is the injected ``expected_error``.
    """
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if not fn.name.startswith("test_"):
            continue
        if not any("parametrize_unsupported(" in ast.unparse(d) for d in fn.decorator_list):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func).endswith("raises")
                and node.args
                and ast.unparse(node.args[0]) == "expected_error"
            ):
                return True
    return False


#: Facet names an argument-fault axis must carry. A kernel takes TENSOR ARGUMENTS, and a caller can
#: get either the dtype or the shape of any of them wrong; both must be refused at the front door
#: with an `API_LEVEL_ERRORS` type, and both must be swept. Measured before this rule existed: an
#: unsupported `mask` dtype raised a bare `KeyError` from the compile-key lookup, an unsupported
#: `layernorm` weight dtype raised an `AssertionError` (which `python -O` strips outright), and no
#: extent of any non-operand tensor was checked at all.
ARG_FAULT_FACETS = ("bad_dtype", "bad_extent")


def coverage_problems(matrix: "KernelMatrix", module_name: str) -> list[str]:
    """Check that a module EXERCISED every axis it declares, covering every facet.

    Purpose
        The gate `Axis.diversity` cannot be. Diversity scores the declared pool, so it catches a
        pool that could not cover a facet -- it cannot catch a pool that could and never did. Every
        narrowing escape hatch (``only=``, ``drop=``, or simply not writing a test for an axis)
        reopens the hole diversity closed, and each is justified by prose that no check reads.

    Semantics
        Two rules, both per MODULE rather than per test. Individual tests stay free to narrow --
        they must, since a K value costs a cold compile here and the pool has five -- but the module
        as a whole may not omit a facet:

        1. Every declared axis is parametrized by at least one test. An axis nothing sweeps is
           decoration: it satisfies the diversity floor and changes nothing that runs.
        2. Every facet of every axis is hit by at least one emitted value, unless the facet carries
           a waiver (``Axis.waived``), which is prose the reader can weigh.

    Args:
        matrix: The module's matrix.
        module_name: The importable module name, matching what `_record_emission` captured. A
            mismatch reads as "nothing emitted" and reports every axis, which is loud rather than
            silent by design.

    Returns:
        A list of problems, empty when the module conforms.
    """
    emitted = _EMITTED.get(matrix.kernel, {}).get(module_name, {})
    problems: list[str] = []
    for axis in matrix.axes:
        seen = emitted.get(axis.name)
        if not seen:
            problems.append(
                f"{module_name}: axis {axis.name!r} is declared but NO test parametrizes it. An "
                f"axis nothing sweeps is decoration -- it satisfies the diversity floor and changes "
                f"nothing that runs. Add a test that draws {axis.name!r} from the matrix, or "
                f"remove the axis."
            )
            continue
        for fname, pred in axis.facets.items():
            if axis.waived.get(fname) or any(pred(v) for v in seen):
                continue
            problems.append(
                f"{module_name}: axis {axis.name!r} declares facet {fname!r} but no test in this "
                f"module ever ran a value satisfying it (ran: {sorted(map(str, seen))}). A pool "
                f"wide enough to pass the diversity check, narrowed away by every test that uses "
                f"it, covers nothing. Widen a test's selection, or waive the facet with a reason."
            )
    return problems


def arg_fault_problems(matrix: "KernelMatrix") -> list[str]:
    """Check that a kernel declares -- and therefore must sweep -- both kinds of argument fault.

    Purpose
        Makes "an unsupported dtype or shape RAISES, and the raise is asserted" a property the
        matrix obliges rather than one each author remembers. It cannot find an argument nobody
        thought to check; what it removes is the cheapness of forgetting, exactly as mandatory
        ``unsupported=`` does for configuration combos.

    Semantics
        Requires one axis carrying every facet in `ARG_FAULT_FACETS`, and requires a region over
        that axis for each -- so the fault is not merely declarable but declared to raise, with a
        message pattern. A kernel that genuinely takes no tensor argument capable of either fault
        waives the facets on the axis, which is prose rather than silence.

    Args:
        matrix: The matrix to check.

    Returns:
        A list of problems, empty when the kernel conforms.
    """
    if matrix.arg_faults_waived:
        return []
    for axis in matrix.axes:
        if set(ARG_FAULT_FACETS) <= set(axis.facets):
            covered = {
                f
                for f in ARG_FAULT_FACETS
                for r in matrix.regions()
                if axis.name in r.axis_names()
                and any(axis.facets[f](v) for v in axis.values if r.where.__doc__ is None)
            }
            missing = [f for f in ARG_FAULT_FACETS if f not in covered and not axis.waived.get(f)]
            if missing:
                return [
                    f"{matrix.kernel}: axis {axis.name!r} carries {missing} but no Unsupported "
                    f"region references it for those, so the fault is declarable and never "
                    f"asserted to raise."
                ]
            return []
    return [
        f"{matrix.kernel}: no axis carries the argument-fault facets {list(ARG_FAULT_FACETS)}. A "
        f"caller can get the dtype OR the shape of any tensor argument wrong, and both must be "
        f"refused at the front door with an API-level error and swept under pytest.raises. Declare "
        f"an axis whose values name those mistakes (see test_dual_gated_gemm.py's 'arg_fault'), or "
        f"waive each facet on it with a reason."
    ]


#: Directories under ``tests/`` whose modules this audit governs, matched on path PARTS at any
#: depth. ``kernels`` and ``workflows`` hold the correctness modules that declare a matrix, ``perf``
#: the gates that must import the same object, ``distributed`` everything the mesh axis reaches.
#:
#: **PARTS, not the immediate parent, and the distinction is the whole point.** A ``parent.name``
#: rule written for ``tests/kernels/`` silently also admits ``tests/distributed/kernels/`` -- by
#: coincidence, because the leaf spelling happens to match -- while refusing its sibling
#: ``tests/distributed/workflows/``. An enforcement that holds by coincidence is indistinguishable
#: from one that holds by decision, and it stops holding the day a directory is renamed. Matching on
#: parts puts both under the audit for the same stated reason, at any depth, and does so **before
#: the directory exists**. Same rule and same spelling as :data:`collective_guard.DIST_DIRS`.
MATRIX_DIRS = frozenset({"kernels", "perf", "workflows", "distributed"})

#: The part that marks a module as a KERNEL module -- one source, one matrix, one perf gate. Narrower
#: than :data:`MATRIX_DIRS` on purpose: ``tests/workflows/`` declares a matrix too, but a workflow
#: composes kernels rather than being one, so the exactly-one-matrix-named-after-the-file rule and
#: the sibling-perf-gate rule do not apply to it.
#:
#: **IT USED TO ALSO GATE `coverage_problems`, WHICH IS A DIFFERENT KIND OF RULE. That is now
#: SPLIT** -- see :func:`_owned_matrices`. Recorded here because the split is the interesting part
#: and this constant is where a reader arrives.
#:
#: Checks 3 and 4 of :func:`audit_test_module` are IDENTITY checks -- is this file THE module for
#: one kernel, does it have a sibling perf gate -- and the narrowing argued above is correct for
#: them. :func:`coverage_problems` asks whether a module SWEPT what it DECLARES, and it had simply
#: inherited this predicate; nobody ever argued the scope. The cost was measured on 2026-08-19:
#: every module under ``tests/distributed/`` -- twelve, all declaring matrices -- got checks 1-2
#: and **NOTHING ELSE**, so a DECORATIVE AXIS PASSED THERE (control: declare an axis, sweep it with
#: a bare ``pytest.mark.parametrize``, 33 collected, audit silent). Two live holes sat in that blind
#: spot: ``test_gemm_a2a_epi.py``'s ``D`` axis declared ``96`` while emitting only ``[4, 8]``, and
#: ``test_gemm_sm90_a2a.py`` could have carried a fabricated axis with no complaint.
#:
#: **Widening this frozenset was NOT the fix, and is still not.** It would drag checks 3 and 4
#: along, demanding a sibling perf gate for every distributed module. Coverage is now scoped on
#: matrix OWNERSHIP instead -- two rules, two predicates.
#:
#: **The general shape, because this happened TWICE from one cause:** an artifact
#: placed one directory deeper than any existing one silently leaves a PATH-BASED auditor's reach,
#: and the auditor reports nothing BECAUSE IT NEVER SEES THE FILE. Silence is the symptom, so no run
#: turns red. The other instance was a perf gate authored at ``tests/distributed/perf/`` while
#: ``test_kernel_matrix.py`` derived its dotted name from ``path.parent.name`` -- see the
#: ``tests_root`` comment there. Ownership scoping is immune by construction; anything still keyed
#: on a path is not.
#:
#: **So when you add a test module or a gate at a NEW depth, check the auditor can still name it.**
#: The cheap check needs no GPU: call :func:`coverage_problems` by hand in a ``--collect-only``
#: session and read the count.
_KERNEL_DIRS = frozenset({"kernels"})


def matrix_scope(path) -> bool:
    """Whether a test module is governed by the matrix audit.

    Args:
        path: Path to a test module, absolute or relative. Matched on the path PARTS, so
            ``tests/distributed/kernels/test_x_a2a.py`` is in scope without this function knowing
            that directory exists. A module whose name does not start with ``test_`` is out of
            scope wherever it sits -- helpers are not audited.

    Returns:
        ``True`` when any path component is in :data:`MATRIX_DIRS` and the file is a ``test_*.py``.
    """
    p = pathlib.Path(path)
    return bool(MATRIX_DIRS & set(p.parts)) and p.name.startswith("test_") and p.suffix == ".py"


def _is_kernel_module(path) -> bool:
    """Whether a module is a per-kernel correctness module, i.e. ``**/kernels/test_<kernel>.py``.

    Args:
        path: Path to the test module, absolute or relative. Read on PARTS, so both
            ``tests/kernels/`` and ``tests/distributed/kernels/`` qualify.

    Returns:
        ``True`` for a ``test_*.py`` under some ``kernels/`` directory. Callers derive the kernel
        name by stripping the ``test_`` prefix, so a False here means checks 3 and 4 of
        :func:`audit_test_module` are skipped entirely rather than applied to a non-kernel file.
    """
    p = pathlib.Path(path)
    return bool(_KERNEL_DIRS & set(p.parts)) and p.name.startswith("test_") and p.suffix == ".py"


def _owned_matrices(tree, module) -> list:
    """The matrices a module CONSTRUCTS, as ``(name, matrix)`` pairs -- the COVERAGE scope.

    Purpose
        Decide where :func:`coverage_problems` applies. Coverage asks "did this module actually
        SWEEP every axis and hit every facet it DECLARES?", which is a property of the module that
        declares the matrix -- not of the directory the file sits in.

    Why ownership and not a directory, which is the whole point of this function
        Coverage used to ride :func:`_is_kernel_module`, i.e. a ``kernels/`` path PART. That
        predicate is argued from IDENTITY (is this file THE module for one kernel, does it have a
        sibling perf gate) and is correct for checks 3 and 4 of :func:`audit_test_module`. Coverage
        is a different KIND of rule and simply inherited it. The consequence was measured on
        2026-08-19: every module under ``tests/distributed/`` -- twelve of them, all declaring
        matrices -- got no coverage check at all, and a DECORATIVE axis passed there silently
        (control: declare an axis, sweep it with a bare ``pytest.mark.parametrize``, 33 collected,
        audit silent). Two live holes had been sitting in that blind spot.

        Ownership cannot be escaped by MOVING A FILE, which is exactly the failure that produced
        this function: an artifact placed one directory deeper than any existing one leaves a
        path-based auditor's reach, and the auditor reports nothing because it never sees the file.
        Silence is the symptom, so no run turns red. A module that declares a matrix is in scope at
        any depth, including depths that do not exist yet.

        ``_KERNEL_DIRS`` is deliberately NOT widened to fix this. Its identity argument is sound,
        and widening it would drag checks 3 and 4 along -- demanding a sibling perf gate for every
        distributed module. Two rules, two predicates.

    Why the AST and not ``vars(module)``, which is the trap this MUST not fall into
        ``vars()`` returns IMPORTED names too, and a perf gate imports the correctness module's
        matrix ON PURPOSE -- the repo REQUIRES it be the same object, so the two cannot drift.
        Auditing coverage there asks "did the perf gate sweep every axis of a matrix it does not
        own", whose answer is legitimately NO: a perf gate pins tiny cells by design. Measured, and
        it is easy to repeat: a ``vars()``-based probe reported **12 problems** against
        ``test_benchmark_perf_dual_gated_gemm_a2a``, every single one an artifact of scanning the
        imported object. An assignment of a ``KernelMatrix(...)`` CALL cannot express an import, so
        reading the AST cannot make that mistake.

    Semantics
        Module-level assignments only (``tree.body``), so a matrix built inside a function -- which
        is how this package's own test file makes throwaway matrices -- is not ownership and is not
        scoped. Matched on the CALL being to something whose dotted name ends in ``KernelMatrix``,
        so an aliased import still matches. EVERY owned matrix is returned, not just the first: the
        previous ``len(found) == 1`` guard silently skipped a module holding two. Measured at
        landing: 21 modules, 21 matrices, one each, 0 problems.

    Args:
        tree: The module's parsed AST, from ``ast.parse`` on its source. Must be the AST of THIS
            module -- another file's tree names variables that do not exist here, every ``getattr``
            below then misses, and the result is a silently EMPTY scope rather than an error.
        module: The IMPORTED module object. Required: the AST gives a NAME, and only the live module
            gives the matrix it is bound to. Callers that cannot import must SKIP the coverage check
            rather than pass a placeholder -- an empty result is indistinguishable from "this module
            declares nothing".

    Returns:
        ``[(name, matrix), ...]`` in source order; empty when the module declares none. A name
        assigned a ``KernelMatrix(...)`` call that is NOT a :class:`KernelMatrix` at runtime (a later
        rebind, a factory returning something else) is dropped rather than reported -- the AST claim
        and the runtime object disagreeing is not a coverage question.
    """
    out = []
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        if not ast.unparse(node.value.func).endswith("KernelMatrix"):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                value = getattr(module, target.id, None)
                if isinstance(value, KernelMatrix):
                    out.append((target.id, value))
    return out


def unsupported_cell_problem(module, params) -> str | None:
    """Report a COLLECTED cell that lands in a declared ``Unsupported`` region, or None.

    Purpose
        Close the gap `KernelMatrix.parametrize` structurally cannot: it evaluates a region only
        when ONE call parametrizes every axis that region reads (``set(region.axis_names()) <=
        set(names)``), because that is all it is given. Stacked decorators defeat that --

            @M.parametrize("N", only=..., because=...)
            @M.parametrize("mesh")

        emits every ``(N, mesh)`` pair while a region reading BOTH is skipped by each call in turn.
        The region is present, correct, and never fires. Measured: four cells declared unsupported
        sat in an accepting test's grid that way.

    Semantics
        Checks the ACTUAL bound cell rather than a predicted cross product, which is why this lives
        at collection and not in the decorator. Consequences, all deliberate:

        * it catches EVERY route to a forbidden combination -- stacked decorators, indirect
          parametrization, a fixture that supplies an axis value -- not just the one that was found;
        * it needs no assumption about decorator order, which the decoration-time form does;
        * an attempt at that form was reverted: ``parametrize`` returns a ``MarkDecorator`` and this
          package's own tests inspect ``.args``, so wrapping the return breaks that contract.

        A cell emitted by :meth:`KernelMatrix.parametrize_unsupported` is EXEMPT and must be -- its
        whole job is to land in a region and assert the raise. It is recognised by the
        ``expected_error`` parameter that method appends, so the exemption is a property of how the
        cell was generated rather than of the test's name.

    EVERY matrix the module holds is checked, not just a lone one
        The question is "does ANY matrix in scope declare this cell unsupported?", and a COUNT is
        not that question. An earlier version bailed on ``len(matrices) != 1``, which made the
        check a silent NO-OP on any module holding two -- exactly what a perf gate looks like once
        it draws some axes from its kernel's matrix and others from a sibling's. Measured on the
        back A2A perf gate, which holds ``A2A_KERNEL`` and ``EPI``: the forbidden cell
        ``{mesh: (('cp',(2,2)),), N_token: 2072, D: 128}`` returned ``None`` there while the same
        cell was correctly REFUSED against the correctness module. Same shape as the
        :func:`_owned_matrices` fix, which likewise replaced a count with a question.

        Zero matrices still yields None, but now because the loop finds nothing rather than
        because a count was special-cased -- there is no state left to guess at.

    ``vars(module)`` is RIGHT here and WRONG in :func:`_owned_matrices`, deliberately
        The two ask opposite questions and therefore need opposite scoping, so do not "harmonise"
        them. Coverage asks *did the module that DECLARES this matrix sweep it?* -- an OWNERSHIP
        question, which is why it parses the AST and ignores imports (a perf gate importing a
        matrix must not be charged with covering it). This function asks *is this bound cell
        forbidden by any matrix the test could have drawn it from?* -- a VISIBILITY question. A
        gate that imports ``A2A_KERNEL`` and parametrizes from it MUST be checked against
        ``A2A_KERNEL``'s regions, and an AST scan for assignments would not see it.

    Args:
        module: The test module object. Every module-level ``KernelMatrix`` it exposes is checked,
            imported ones included; a module with none is simply not flagged.
        params: The item's bound parameters, i.e. ``item.callspec.params``. Keys that are not
            declared axes are ignored, so an ordinary fixture argument cannot trip this.

    Returns:
        A human-readable problem naming the cell and the region, or None when the cell is legal.
        Never raises: an un-evaluable region (a ``where`` that throws on these values) is skipped,
        because failing a test for the guard's own bug would be worse than missing one cell.
    """
    if "expected_error" in params:
        return None
    # Deduped BY IDENTITY, not by name: one matrix bound to two names (an alias, or a re-export)
    # is one matrix, and checking it twice would make the reported message depend on which name
    # happened to come first in `vars()`.
    matrices, seen = [], set()
    for v in vars(module).values():
        if isinstance(v, KernelMatrix) and id(v) not in seen:
            seen.add(id(v))
            matrices.append(v)
    for matrix in matrices:
        bound = {a.name: params[a.name] for a in matrix.axes if a.name in params}
        for region in matrix.regions():
            axes = set(region.axis_names())
            if not axes <= set(bound):
                continue
            with contextlib.suppress(Exception):
                if region.where(**{k: bound[k] for k in axes}):
                    return (
                        f"{matrix.kernel}: the collected cell "
                        f"{ {k: bound[k] for k in sorted(axes)} } lies in the unsupported region "
                        f"{region.reason!r}, which must RAISE -- but it was emitted by a test that "
                        f"asserts SUCCESS. `parametrize` could not catch this because no single call "
                        f"parametrizes all of {sorted(axes)}; separate decorators are CROSSED by "
                        f"pytest. Parametrize those axes JOINTLY "
                        f"({matrix.kernel}.parametrize({', '.join(repr(a) for a in sorted(axes))}, "
                        f"cells=[...], because='...')), and cover the refusal with "
                        f"parametrize_unsupported()."
                    )
    return None


def audit_test_module(path, module=None) -> list[str]:
    """Check one test module against the matrix rules. Returns problems; does not raise.

    The single implementation behind both enforcement points: ``tests/conftest.py`` calls it at
    **collection** (so it cannot be skipped by running a subset) and
    ``tests/testing/test_kernel_matrix.py`` calls it as a test (so failures read as test failures
    with a full report). One function, so the two can never disagree.

    Checks applied, in order:

    1. Every ``def test_*`` either parametrizes from a matrix or carries ``@matrix_exempt``. A bare
       ``pytest.mark.parametrize`` does **not** satisfy this -- it is exactly the hand-rolled list
       the matrix replaces. A module-level ``pytestmark = [matrix_exempt("...")]`` satisfies it for
       EVERY test in the module, for a module whose subject is not a kernel at all; its reason is
       held to the same literal-string rule as a per-test one.
    2. Every exemption's reason is a non-empty **string literal** at the decoration site. It is read
       from the AST, so an f-string or a name reads as no reason.
    3. If the module is ``**/kernels/test_<k>.py`` (see :func:`_is_kernel_module` -- matched on path
       PARTS, so ``tests/distributed/kernels/`` counts): it defines exactly one module-level
       :class:`KernelMatrix`, whose ``kernel`` is ``<k>``, and there is a perf gate in the SIBLING
       ``perf/`` directory. Requires ``module``.
    4. If the module is ``test_benchmark_perf_<k>.py``: it holds a module-level reference to the
       *same object* the correctness module defines -- not a copy, not a second matrix. Requires
       ``module``.

    Args:
        path: Path to the test module. Its PARTS decide which of 3/4 apply, so it must be the real
            location, not a temporary copy under a different name or at a different depth.
        module: The imported module object, or None to run only the AST checks (1 and 2). Passing
            None where 3/4 apply silently skips them, so callers that can import should.

    Returns:
        A list of human-readable problems, empty when the module conforms. Each entry names the
        module and what to do about it.
    """
    import importlib
    import pathlib

    path = pathlib.Path(path)
    src = path.read_text()
    problems: list[str] = []
    tree = ast.parse(src)

    # A module-level ``pytestmark = [matrix_exempt("...")]`` exempts EVERY test in the module. It
    # exists for a module whose SUBJECT is not a kernel at all -- a ported harness whose tests drive a
    # cell protocol with no shapes -- where the same sentence would otherwise be repeated once per
    # test. Repeating it 45 times does not make a reader more likely to disagree with it; it makes the
    # file harder to diff against the tree it was ported from, which is a cost with no matching
    # benefit. The reason is validated EXACTLY as the per-test one is: a non-empty string literal read
    # from the AST, so an f-string or a name still reads as no reason.
    module_exempts = [
        c
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
        for c in ast.walk(node.value)
        if isinstance(c, ast.Call) and ast.unparse(c.func).endswith("matrix_exempt")
    ]
    module_exempt = False
    if module_exempts:
        marg = module_exempts[0].args[0] if module_exempts[0].args else None
        if isinstance(marg, ast.Constant) and isinstance(marg.value, str) and marg.value.strip():
            module_exempt = True
        else:
            problems.append(
                f"{path.name}: the module-level pytestmark matrix_exempt needs a non-empty string "
                f"LITERAL reason -- the AST is what is read, so an f-string or a name is no reason."
            )

    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if not fn.name.startswith("test_") or module_exempt:
            continue
        decs = [ast.unparse(d) for d in fn.decorator_list]
        matrix_driven = (".parametrize(", ".parametrize_unsupported(")
        if any(
            any(tag in d for tag in matrix_driven) and not d.startswith("pytest.mark.parametrize")
            for d in decs
        ):
            continue
        exempts = [
            d
            for d in fn.decorator_list
            if isinstance(d, ast.Call) and ast.unparse(d.func).endswith("matrix_exempt")
        ]
        if not exempts:
            problems.append(
                f"{path.name}::{fn.name} neither parametrizes from a KernelMatrix nor declares "
                f'@matrix_exempt("why the matrix does not apply"). A bare pytest.mark.parametrize '
                f"is the hand-rolled list the matrix exists to replace."
            )
            continue
        arg = exempts[0].args[0] if exempts[0].args else None
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.strip()):
            problems.append(
                f"{path.name}::{fn.name} is @matrix_exempt but its reason is not a non-empty "
                f"string literal; it is read from the AST, so it must be legible in the source."
            )

    name = path.name
    if module is not None and _is_kernel_module(path):
        kernel = name[len("test_") : -len(".py")]
        found = [v for v in vars(module).values() if isinstance(v, KernelMatrix)]
        if len(found) != 1:
            problems.append(
                f"{name} must define exactly ONE module-level KernelMatrix for kernel "
                f"{kernel!r}; found {len(found)}. The matrix lives beside the tests it governs, "
                f"so a kernel cannot be tested without declaring its configs."
            )
        elif found[0].kernel != kernel:
            problems.append(
                f"{name} declares a KernelMatrix for {found[0].kernel!r}, but its filename says "
                f"{kernel!r}. The perf gate looks it up by filename, so these must agree."
            )
        elif found[0].regions() and not _has_working_unsupported_test(ast.parse(src)):
            problems.append(
                f"{name} declares {len(found[0].regions())} unsupported region(s) but has no test "
                f"that DECORATES with parametrize_unsupported() and asserts the raise inside "
                f"pytest.raises(expected_error, match=expected_match). A region that is never "
                f"exercised is decoration: the kernel could stop refusing the combo and nothing "
                f"would notice. Regions: {[r.reason for r in found[0].regions()]}"
            )
        elif arg_fault_problems(found[0]):
            problems.extend(arg_fault_problems(found[0]))
        elif not (path.parent.parent / "perf" / f"test_benchmark_perf_{kernel}.py").exists():
            # The perf gate is the SIBLING `perf/` of the kernels directory, computed rather than
            # hardcoded to `tests/perf/`. That generalization is load-bearing for the distributed
            # tree: `tests/distributed/kernels/` gets `tests/distributed/perf/`, which is correct
            # and not merely convenient -- a distributed gate has to be launched under
            # torchrun/srun, so it cannot sit in `tests/perf/`, which runs in a plain isolated
            # session and would collect it with no process group.
            gate_parts = (path.parent.parent / "perf" / f"test_benchmark_perf_{kernel}.py").parts
            i = (
                len(gate_parts) - 1 - gate_parts[::-1].index("tests")
                if "tests" in gate_parts
                else 0
            )
            gate = "/".join(gate_parts[i:])
            problems.append(
                f"{name} declares a matrix but there is no {gate}. A kernel's speed is part of "
                f"its contract here; add the perf gate, drawing on the same matrix."
            )

    # COVERAGE is scoped on OWNERSHIP, not on `_is_kernel_module`. The two checks above ask an
    # IDENTITY question and are correctly gated on a `kernels/` path part; this one asks whether the
    # module SWEPT what it DECLARES, which is true of any module declaring a matrix at any depth.
    # See `_owned_matrices` for the measurement behind the split, and for why `vars(module)` is the
    # wrong selector (it reports 12 phantom problems against a perf gate).
    if module is not None:
        for _name, _matrix in _owned_matrices(tree, module):
            problems.extend(coverage_problems(_matrix, getattr(module, "__name__", "?")))

    if module is not None and name.startswith("test_benchmark_perf_"):
        kernel = name[len("test_benchmark_perf_") : -len(".py")]
        # The owning module is the one that MIRRORS ITS SOURCE, and not every subject lives under
        # `fold_cp_ops/kernels/` -- `workflows/trimul_autotune.py` is owned by
        # `tests/workflows/test_trimul_autotune.py`. So the matrix owner is SEARCHED FOR, not enumerated:
        # an enumeration has to be edited every time a source moves, and the two times that was
        # missed the perf gate silently stopped comparing against any matrix at all. A single
        # `test_<kernel>.py` anywhere under `tests/` is exactly what the naming rule guarantees, so
        # finding it is well-defined; finding two is the basename collision the rule forbids, and
        # is reported rather than resolved by picking one.
        # The tests ROOT, not "one directory up". `tests/perf/` makes those the same thing and a
        # nested gate directory does not: for `tests/distributed/perf/`, `path.parent.parent` is
        # `tests/distributed`, so the dotted name below came out `distributed.test_x` and the
        # import failed with "No module named 'distributed'" -- reported as a missing `__init__.py`
        # that is in fact present. Anchoring on the nearest ancestor named `tests` is identical for
        # `tests/perf/` and correct at any depth. The fallback keeps today's behaviour for a layout
        # that has no such ancestor rather than inventing one.
        tests_root = next((q for q in path.parents if q.name == "tests"), path.parent.parent)
        hits = sorted(tests_root.rglob(f"test_{kernel}.py"))
        owner = None
        if len(hits) == 1:
            dotted = ".".join(
                (tests_root.name, *hits[0].relative_to(tests_root).with_suffix("").parts)
            )
            try:
                owner = importlib.import_module(dotted)
            except ImportError as exc:
                problems.append(
                    f"{name}: found the owning module at {hits[0]} but importing it as {dotted} "
                    f"failed ({exc}). Every directory on that path needs an __init__.py."
                )
        elif not hits:
            problems.append(
                f"{name}: cannot find the owning matrix module to compare; searched "
                f"{tests_root}/**/test_{kernel}.py. The perf gate's subject must have a test "
                f"module at the path mirroring its source."
            )
        else:
            problems.append(
                f"{name}: found {len(hits)} modules named test_{kernel}.py ({', '.join(str(h) for h in hits)}). "
                f"A source's coverage lives in exactly ONE file, so which one owns the matrix is "
                f"undefined; rename the SOURCE files so their basenames differ."
            )
        if owner is not None:
            declared = [v for v in vars(owner).values() if isinstance(v, KernelMatrix)]
            mine = [v for v in vars(module).values() if isinstance(v, KernelMatrix)]
            if declared and not any(m is declared[0] for m in mine):
                problems.append(
                    f"{name} must IMPORT the KernelMatrix from {owner.__name__} "
                    f"(`from {owner.__name__} import ...`), not define its own. Timed "
                    f"and correctness-tested shapes have to come from one declaration or they "
                    f"drift apart."
                )
    return problems


def matrix_exempt(reason: str):
    """Mark a test as legitimately outside the kernel matrix. **The reason is mandatory.**

    For tests whose subject is not a kernel launch over shapes -- a pure helper function, an input
    validation path, a cache-key property. Without this, the lock in
    ``tests/testing/test_kernel_matrix.py`` fails the test for not drawing on a matrix; with it, the
    exemption is a sentence in the source that a reviewer can disagree with.

    Args:
        reason: Why the matrix does not apply. Must be a non-empty **string literal at the
            decoration site** -- the lock reads it from the AST, so an f-string or a name lookup
            reads as no reason at all and is rejected there.

    Returns:
        A pytest marker. Purely declarative: it changes nothing about how the test runs.

    Raises:
        ValueError: If ``reason`` is empty or whitespace.
    """
    if not (reason and reason.strip()):
        raise ValueError("matrix_exempt() requires a non-empty reason")
    return pytest.mark.matrix_exempt(reason)
