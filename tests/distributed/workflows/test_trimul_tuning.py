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



"""Unit tests for `fold_cp_ops.distributed.workflows.trimul_tuning`.

CPU-only and GPU-free by construction: the subject is a dataclass tree and a name mapping, and the
one thing that could need hardware -- that the emitted names are real -- is answered by
`inspect.signature`, not by a launch.
"""

from __future__ import annotations

import inspect

import pytest

from fold_cp_ops.distributed.workflows.trimul_tuning import (
    NOT_EXPOSED,
    BackTuning,
    FrontTuning,
    OutGateTuning,
    TransportTuning,
    TriMulTuning,
)
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

#: The numeric coverage gate asks whether a test that RAN A KERNEL made a sanctioned comparison.
#: Nothing here runs one: the subject is a dataclass tree and a name mapping, and the assertions are
#: over dicts and name sets. A per-test `@numeric_exempt` is not enough -- with no
#: matrix-parametrized test in the module the gate has nothing to attach to, so the declaration has
#: to be module-level.
NUMERIC_EXEMPT = (
    "pure configuration-vocabulary module: it constructs no tensor, launches no kernel and "
    "compares no numerical result -- every assertion is over a dict or a set of parameter names"
)

pytestmark = matrix_exempt(
    "the subject is a dataclass tree and a name mapping -- there is no kernel, no shape and no "
    "device, so there is nothing for a KernelMatrix to sweep"
)

_GROUPS = (FrontTuning, BackTuning, OutGateTuning, TransportTuning)


def _engine_kwonly() -> set[str]:
    """The engine constructor's keyword-only parameter names, read from the live signature."""
    from fold_cp_ops.distributed.workflows.trimul_autotuned import TriMulAutotuned

    sig = inspect.signature(TriMulAutotuned.__init__)
    return {n for n, p in sig.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY}


def test_a_default_tuning_emits_no_engine_kwargs_at_all():
    """`TriMulTuning()` must flatten to `{}` -- the property the whole design rests on.

    A default-tuned module has to call the engine with LITERALLY the arguments it is called with
    today, not with the same values re-spelled. That is what makes argument identity a property of
    the type rather than something to re-measure, and it is what the R0 oracle
    (`trimul_engine_args.json`) is diffed against.

    Defaulting a field to the engine's own default instead of to ``None`` would break this while
    looking harmless: every call would then carry an explicit keyword the old call never mentioned,
    every oracle cell would differ, and the diff would be noise rather than signal.
    """
    assert TriMulTuning().to_engine_kwargs() == {}
    for group in _GROUPS:
        assert all(getattr(group(), f.name) is None for f in group.__dataclass_fields__.values()), (
            f"{group.__name__} has a field that does not default to None"
        )


def test_every_emitted_key_is_a_real_engine_parameter():
    """Every keyword this module can emit must exist on `TriMulAutotuned.__init__`.

    This is the check that makes the mapping fail LOUDLY and EARLY. Without it a knob renamed on
    the engine leaves this module emitting a keyword nobody accepts, and the failure surfaces as a
    `TypeError` from a constructor the caller never named -- on the first forward, on a cluster.
    """
    engine = _engine_kwonly()
    emitted = {k for g in _GROUPS for k in g._ENGINE_KEYS.values()} | set(
        TriMulTuning._ENGINE_KEYS.values()
    )
    unknown = sorted(emitted - engine)
    assert not unknown, (
        f"these tuning keys are not parameters of TriMulAutotuned.__init__: {unknown}. Either the "
        f"engine renamed them or this mapping invented them; both are silent until a forward."
    )


def test_every_engine_knob_is_either_exposed_or_explicitly_not():
    """Partition the engine's keyword-only surface: exposed, or listed in `NOT_EXPOSED` with a why.

    The point is not coverage for its own sake. An engine knob that is NEITHER exposed NOR declared
    unexposed is indistinguishable from one nobody thought about -- which is exactly the state
    `**fused_kwargs` left every knob in. Adding a parameter to the engine now fails here until
    somebody decides which side it is on.
    """
    engine = _engine_kwonly()
    exposed = {k for g in _GROUPS for k in g._ENGINE_KEYS.values()} | set(
        TriMulTuning._ENGINE_KEYS.values()
    )
    undecided = sorted(engine - exposed - set(NOT_EXPOSED))
    assert not undecided, (
        f"engine knobs that are neither exposed nor in NOT_EXPOSED: {undecided}. Add a field, or "
        f"add an entry saying why not -- 'we know it is not tunable' and 'it is written down as "
        f"not tunable' must not be the same state."
    )
    stale = sorted(set(NOT_EXPOSED) - engine)
    assert not stale, f"NOT_EXPOSED names knobs the engine no longer has: {stale}"
    overlap = sorted(exposed & set(NOT_EXPOSED))
    assert not overlap, f"these are both exposed and declared unexposed: {overlap}"


def test_a_set_knob_reaches_the_engine_under_the_engine_s_own_name():
    """A value that IS set must arrive under the engine's spelling, and only that value."""
    t = TriMulTuning(
        front=FrontTuning(tile_mn=(128, 64)),
        back=BackTuning(cluster_n=4),
        out_gate=OutGateTuning(consumer="staged_a_in_regs"),
        autotune=True,
    )
    assert t.to_engine_kwargs() == {
        "front_tile_mn": (128, 64),
        "back_cluster_n": 4,
        "consumer": "staged_a_in_regs",
        "autotune_config": True,
    }


def test_an_unknown_field_is_refused_at_construction():
    """A typo must raise HERE, naming the field -- not three frames into a forward on a cluster.

    This is the whole delta against `**fused_kwargs`, which accepts any spelling and defers the
    complaint to `TriMulAutotuned.__init__` at first use.
    """
    with pytest.raises(TypeError, match="tile_mm"):
        FrontTuning(tile_mm=(128, 128))


def test_the_groups_are_frozen():
    """Tuning is read at construction; a later mutation would be silently ignored, so refuse it."""
    f = FrontTuning(tile_mn=(128, 128))
    with pytest.raises(Exception):
        f.tile_mn = (256, 128)
