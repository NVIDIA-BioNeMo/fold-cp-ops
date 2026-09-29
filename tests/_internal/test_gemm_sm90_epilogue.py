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
"""Tests for ``fold_cp_ops._internal.gemm_sm90_epilogue`` -- the store layer, in ISOLATION.

The mirror of ``test_gemm_sm90_load.py``. That file proves the transport INTO the accumulator; this
one proves the transport OUT of it, with the MMA's arithmetic removed from the question entirely.

**How the isolation works, and why it costs one overridden hook rather than a harness kernel.**
``epi_visit_acc`` is an existing seam that runs *after* ``mma`` and *before* ``epilogue``. The probe
overrides it to **overwrite the accumulator with a value derived from each element's own output
coordinate**, then lets the real epilogue run untouched. So:

* the mainloop, the pipeline and the MMA all still execute -- nothing deadlocks, and the probe does
  not have to reimplement the producer/consumer handshake the way the load probe did;
* whatever the MMA computed is discarded, so the operands are irrelevant and the arithmetic is out
  of scope;
* the expected output is known **exactly and per element**: ``D[i, j] == i * tile_N + j``.

That last point is what makes this strong. A constant fill would pass even if the retile duplicated
one lane's registers across four; a coordinate-derived fill cannot. Every register in the
accumulator fragment carries its own address, so any permutation, drop or duplication anywhere in
``epi_retile_acc`` -> ``epilog_smem_store_and_partition`` -> the r2s copy -> SMEM -> the TMA store
lands as a wrong value at a named coordinate.

The values are small integers (max ``tile_M * tile_N`` = 16384 at the default tile, far under
fp32's 2**24) and D is fp32, so every assertion is ``assert_bitwise`` -- the epilogue is transport
plus a dtype conversion, and neither is allowed a tolerance.
"""

import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from fold_cp_ops._internal.arch import get_max_active_clusters
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_scheduler_args,
    perm3d,
)
from fold_cp_ops.kernels.gemm_sm90 import GemmSm90
from fold_cp_ops.testing.numerics import assert_bitwise, integer_operands

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(_SM != 9, reason=f"needs sm_90; this GPU is sm_{_SM}0")


class _EpilogueProbeGemm(GemmSm90):
    """A ``GemmSm90`` that replaces the accumulator with a coordinate ramp before the epilogue.

    Only ``epi_visit_acc`` is overridden -- the smallest seam that reaches the accumulator after the
    mainloop has finished with it. Everything else is the production path, including the epilogue
    under test.

    Input requirements (unchecked; a violation makes the expectation wrong, not the kernel):

    * ``mma_warp_groups == 1`` and no pingpong, i.e. ``tile_M <= 128``. The coordinate tensor is
      partitioned with ``tiled_mma.get_slice(0)``, which matches how the real role slices the MMA
      only when there is a single MMA warpgroup. With two, each warpgroup owns a different M half
      and the slice index would have to follow.
    * a single work tile (``M <= tile_M``, ``N <= tile_N``), so the ramp's coordinates are the
      output's own coordinates rather than a tile-local offset.
    * an fp32 output, so the ramp survives the store bit-exactly.
    """

    @cute.jit
    def epi_visit_acc(self, params, acc, tiled_mma, tile_coord_mnkl, tidx):
        """Overwrite every accumulator register with ``m * tile_N + n`` for its own coordinate.

        ``partition_C`` of an identity tensor gives each thread the (m, n) coordinate of each of its
        accumulator registers, in the *same* fragment layout as ``acc`` -- so a linear walk over the
        two in step pairs each register with its address. That pairing is the whole trick: it is
        what turns "the epilogue stored something" into "the epilogue stored element (i, j) at
        (i, j)".

        Args:
            params: Epilogue params. Unused.
            acc: The accumulator fragment, **overwritten in place**.
            tiled_mma: The MMA atom, used to partition the coordinate tensor identically to ``acc``.
            tile_coord_mnkl: This work tile's coordinate. Unused -- the probe runs one tile.
            tidx: Thread index within the warpgroup. **Load-bearing** -- it selects the MMA
                slice whose ``partition_C`` gives THIS thread's coordinates. Slicing at 0 instead
                yields an identically-SHAPED tensor holding thread 0's coordinates, which compiles,
                runs, and is wrong.

        Returns:
            None. ``acc`` is modified in place, which is what the real hook's contract allows.
        """
        tile_M = const_expr(self.cta_tile_shape_mnk[0])
        tile_N = const_expr(self.cta_tile_shape_mnk[1])
        # get_slice(TIDX), not get_slice(0). The fragment SHAPE is the same either way --
        # which is why a wrong slice compiles and runs -- but the coordinate VALUES are the
        # sliced thread's. With get_slice(0) every thread wrote thread 0's coordinates, and
        # the ramp came out as 0,1,0,1,... across the row.
        thr_mma = tiled_mma.get_slice(tidx)
        cD = thr_mma.partition_C(cute.make_identity_tensor((tile_M, tile_N)))
        # NO filter_zeros here. It collapses stride-0 modes, and acc and cD do not necessarily
        # have the same ones -- filtering both independently made index i mean different elements
        # in the two, which showed up as the ramp landing two columns late. Walking the UNFILTERED
        # fragments keeps `acc[i]` and `cD[i]` the same register by construction.
        for i in cutlass.range_constexpr(cute.size(acc)):
            m, n = cD[i]
            acc[i] = Float32(m * tile_N + n)


_COMPILED = {}


def _run_epilogue_probe(A, B, D, tile_M, tile_N):
    """Compile (once per config) and launch the epilogue probe.

    Args:
        A: A operand. Its VALUES are irrelevant -- the accumulator they produce is overwritten --
            but it must exist so the real mainloop runs and the pipeline stays healthy.
        B: B operand, same.
        D: fp32 output, ``(1, M, N)``, **written in place**.
        tile_M: CTA tile M. Must be <= 128 (one MMA warpgroup) and >= D's M.
        tile_N: CTA tile N. Must be >= D's N.

    Returns:
        None.
    """
    a_dt, d_dt = torch2cute_dtype_map[A.dtype], torch2cute_dtype_map[D.dtype]
    A_p, B_p, D_p, _ = perm3d(A, B, D, None)
    key = (a_dt, d_dt, tile_M, tile_N)
    if key not in _COMPILED:
        mA, mB, mD, mC, _, _, _, l = make_fake_gemm_tensors(
            a_dt, a_dt, d_dt, None, "k", "k", "n", None
        )
        _COMPILED[key] = compile_gemm_kernel(
            _EpilogueProbeGemm,
            a_dt,
            (tile_M, tile_N),
            (1, 1, 1),
            False,
            True,
            False,
            (9, 0),
            mA,
            mB,
            mD,
            mC,
            (),
            make_fake_scheduler_args(False, False, l),
        )
    clusters = get_max_active_clusters(1)
    _COMPILED[key](A_p, B_p, D_p, None, (), make_scheduler_args(clusters, 8, None, None))


def _ramp(M, N, tile_N):
    """The reference the probe must reproduce: ``ref[i, j] = i * tile_N + j``.

    Args:
        M: Rows to build.
        N: Columns to build.
        tile_N: The CTA tile's N extent, which is the ramp's stride -- **not** N. The accumulator is
            addressed in TILE coordinates, so a ramp built with N would disagree wherever
            ``N != tile_N`` and the test would fail for its own reason.

    Returns:
        An fp32 CUDA tensor of shape ``(1, M, N)``.
    """
    i = torch.arange(M, device="cuda", dtype=torch.float32).unsqueeze(1)
    j = torch.arange(N, device="cuda", dtype=torch.float32).unsqueeze(0)
    return (i * tile_N + j).unsqueeze(0)


@requires_sm90
@pytest.mark.parametrize(
    "M,N,tile_M,tile_N",
    [
        (128, 128, 128, 128),  # the tile exactly
        (128, 256, 128, 256),  # a wider tile: more epilogue subtiles per work tile
        (64, 64, 64, 64),  # a narrower tile
        (65, 200, 128, 256),  # BOTH extents off the tile -- the ragged store
        (1, 8, 128, 128),  # a single row, deep in a mostly-masked tile
    ],
)
def test_the_epilogue_stores_every_accumulator_register_at_its_own_coordinate(M, N, tile_M, tile_N):
    """``D[i, j] == i * tile_N + j``, bit for bit, for a coordinate ramp planted in the accumulator.

    The isolated statement about the store path: the MMA's arithmetic is discarded, so a failure
    here is the retile, the r2s partition, the SMEM staging or the TMA store -- and the failing
    coordinate says which element went astray.

    The last two cells matter most. ``(65, 200)`` puts both extents off the tile grid, so most of
    the accumulator is masked and only the live part may be stored; ``(1, 8)`` is the degenerate
    version of the same thing. An over-wide store predicate writes ramp values into memory the
    caller did not ask for, and an over-tight one leaves elements unwritten -- the comparison is
    against a poisoned buffer, so both are caught rather than only the second.
    """
    g = torch.Generator(device="cuda").manual_seed(21)
    A = integer_operands((1, M, 64), torch.bfloat16, "cuda", generator=g, bound=2)
    B = integer_operands((1, N, 64), torch.bfloat16, "cuda", generator=g, bound=2)
    # Poison the output: anything the epilogue does not store keeps this and fails the comparison.
    D = torch.full((1, M, N), -7777.0, device="cuda", dtype=torch.float32)

    _run_epilogue_probe(A, B, D, tile_M, tile_N)

    assert_bitwise(
        D,
        _ramp(M, N, tile_N),
        what=f"epilogue coordinate ramp (M={M} N={N} tile=({tile_M},{tile_N}))",
    )


@requires_sm90
def test_the_epilogue_does_not_write_outside_the_requested_extent():
    """A ragged tile stores its live elements and **nothing** past them.

    The complement of the ramp test: that one checks every requested element is right, this one
    checks no unrequested element is touched. Allocate a wider buffer, hand the kernel a narrow
    view of it, and require the surrounding columns to keep their poison.

    This is the failure a same-shaped comparison structurally cannot see -- an epilogue that stores
    a full ``tile_N``-wide subtile into a narrower output writes past the view, and every element
    the test looks at is still correct.
    """
    M, N, pad = 65, 200, 256
    g = torch.Generator(device="cuda").manual_seed(22)
    A = integer_operands((1, M, 64), torch.bfloat16, "cuda", generator=g, bound=2)
    B = integer_operands((1, N, 64), torch.bfloat16, "cuda", generator=g, bound=2)

    backing = torch.full((1, M, pad), -7777.0, device="cuda", dtype=torch.float32)
    D = backing[:, :, :N]
    _run_epilogue_probe(A, B, D, 128, 256)

    assert_bitwise(D, _ramp(M, N, 256), what="the live extent")
    tail = backing[:, :, N:]
    assert_bitwise(
        tail,
        torch.full_like(tail, -7777.0),
        what=f"columns {N}..{pad} beyond the requested extent (must be untouched)",
    )


@requires_sm90
def test_the_ramp_is_reproducible_across_launches():
    """Repeating the launch gives the same bytes.

    The epilogue stages through a small ring of SMEM buffers and a store pipeline; a missing
    producer/consumer arrival shows up as a subtile that is *sometimes* the previous one's. Ten
    launches compared against the first turn that into a direct failure instead of a flaky one.

    Ten is enough to be useful and cheap. It is not a proof of absence and is not claimed as one.
    """
    g = torch.Generator(device="cuda").manual_seed(23)
    A = integer_operands((1, 128, 128), torch.bfloat16, "cuda", generator=g, bound=2)
    B = integer_operands((1, 128, 128), torch.bfloat16, "cuda", generator=g, bound=2)

    first = torch.full((1, 128, 128), -7777.0, device="cuda", dtype=torch.float32)
    _run_epilogue_probe(A, B, first, 128, 128)
    for i in range(9):
        again = torch.full((1, 128, 128), -7777.0, device="cuda", dtype=torch.float32)
        _run_epilogue_probe(A, B, again, 128, 128)
        assert_bitwise(again, first, what=f"epilogue store, launch {i + 2} of 10")


def test_the_epilogue_visit_hooks_default_to_doing_nothing():
    """The base class's ``epi_*`` hooks are trivial, and each in its own specific way.

    They exist so a bare ``GemmSm90`` is concrete and emits exactly ONE store. Three of them were
    absent upstream -- every mixin happened to supply them, so the base died on ``AttributeError``
    several frames inside ``epilogue()``. Asserting the defaults here is what keeps a future edit
    from quietly making the base abstract again.

    ``epi_visit_subtile`` is checked by identity rather than equality: a default that rebuilt an
    equal tensor would be per-subtile work every plain GEMM pays for nothing.
    """
    from fold_cp_ops._internal.gemm_sm90_epilogue import GemmSm90EpilogueMixin as E

    sentinel = object()
    # epi_visit_subtile modifies tRS_rD IN PLACE and RETURNS the optional post-activation
    # fragment. None therefore means "no second store", which is what makes a plain GEMM emit
    # exactly one -- it is not an oversight, and asserting it returns its input would be wrong.
    assert E.epi_visit_subtile(None, None, None, None) is None
    assert E.epi_setup_postact(None, None, None, None, None, None, None) is None
    # epi_convert_postact IS the identity: unreachable unless epi_setup_postact returned non-None.
    assert E.epi_convert_postact(None, sentinel, None, None, None, None, None) is sentinel
    assert E.maybe_override_epi_tile(None, sentinel) is sentinel


# ── the extracted seam: EpiPartition + epilogue_partition ─────────────────────────────────────
def test_epi_partition_binds_by_keyword_and_refuses_a_typo():
    """The bundle is keyword-only and slot-checked.

    Four of its nine fields are same-typed register fragments (``tRS_rD``, ``tRS_rC``, ``tSR_rC``,
    ``tRS_rAcc``), so a positional constructor would let two be swapped with no diagnostic -- the
    epilogue would store the C addend where the output belongs and produce plausible numbers.
    """
    from fold_cp_ops._internal.gemm_sm90_epilogue import EpiPartition

    full = dict.fromkeys(EpiPartition.__slots__, 0)
    e = EpiPartition(**full)
    assert e.tRS_rD == 0
    with pytest.raises(TypeError, match=r"missing field"):
        EpiPartition(tRS_rD=1)
    with pytest.raises(AttributeError):
        EpiPartition(**full, tRS_rDD=1)


def test_epilogue_partition_is_a_plain_def_and_the_role_uses_it():
    """The seam must stay an undecorated ``def``, and the role must go through it.

    ``@cute.jit`` flattens arguments into MLIR values and cannot return an :class:`EpiPartition`,
    so decorating this would break every derivation that calls it -- while the in-tree tests, which
    only exercise the base, stayed green. And if the role re-inlines the partitioning, overrides
    become dead code with the same silence. Both are source-level facts, so both are checked from
    source.

    "The role" is ``mma_warpgroup_role`` PLUS ``run_epilogue``, the extract-method it delegates the
    per-work-tile store to. The two are checked together rather than separately because which of
    them holds the call is not the property: a derivation that dispatches between two epilogues per
    work tile overrides ``run_epilogue``, and moving the partitioning down into it is what made that
    possible. What must stay true is that SOMETHING on the base path calls the seam, and that
    nothing on it reaches past the seam to the layer below.
    """
    import inspect

    from fold_cp_ops._internal.gemm_sm90_epilogue import GemmSm90EpilogueMixin
    from fold_cp_ops._internal.gemm_sm90_mma import GemmSm90MmaMixin

    fn = GemmSm90EpilogueMixin.epilogue_partition
    assert not hasattr(fn, "__wrapped__"), "a @cute.jit seam cannot return an EpiPartition"
    src = inspect.getsource(GemmSm90MmaMixin.mma_warpgroup_role) + inspect.getsource(
        GemmSm90MmaMixin.run_epilogue
    )
    assert "self.epilogue_partition(" in src
    assert "epilog_smem_store_and_partition" not in src, (
        "the role must partition through the seam, or overrides are dead code"
    )


@requires_sm90
def test_epilogue_partition_is_re_entrant_within_one_work_tile():
    """Calling it more than once per work tile changes nothing.

    This is the property the dual-gated kernel depends on: it produces several accumulators per
    work tile (one per output-N sub-tile) and partitions once for each. If partitioning had a side
    effect -- advancing a pipeline, consuming an SMEM buffer, mutating ``acc`` -- the second pass
    would corrupt the first.

    Tested by a probe that calls the seam an EXTRA time and discards the result before doing the
    real work. Output must be bit-identical to the unmodified kernel.
    """
    from fold_cp_ops.kernels.gemm import gemm
    from fold_cp_ops.kernels.gemm_sm90 import GemmSm90 as _Base

    class _DoublePartition(_Base):
        """Partitions twice per work tile, keeping the second bundle."""

        def epilogue_partition(self, tiled_mma, acc, sD, sC, tidx, has_C):
            """Call the real seam twice and return the second bundle."""
            super().epilogue_partition(tiled_mma, acc, sD, sC, tidx, has_C)
            return super().epilogue_partition(tiled_mma, acc, sD, sC, tidx, has_C)

    M, N, K = 384, 512, 256
    g = torch.Generator(device="cuda").manual_seed(51)
    A = integer_operands((1, M, K), torch.float16, "cuda", generator=g, bound=64)
    B = integer_operands((1, N, K), torch.float16, "cuda", generator=g, bound=64)

    base_out = torch.zeros(1, M, N, device="cuda", dtype=torch.float32)
    gemm(A, B, base_out, None, None, 128, 128, 1, 1)

    twice = torch.zeros(1, M, N, device="cuda", dtype=torch.float32)
    _run_probe(_DoublePartition, A, B, twice, 128, 128)
    assert_bitwise(twice, base_out, what="partitioning twice per work tile")


@requires_sm90
def test_the_retiled_accumulator_is_a_view_of_acc_not_a_copy():
    """``tRS_rAcc`` aliases ``acc``'s registers, so a hook may still rewrite them after partitioning.

    The role partitions BEFORE calling ``epi_visit_acc``, so if ``epi_retile_acc`` copied rather
    than re-laid-out, every hook that writes the accumulator -- the gated kernels' activation
    fusion among them -- would be silently discarded and the epilogue would store the pre-hook
    values.

    ``test_the_epilogue_stores_every_accumulator_register_at_its_own_coordinate`` above is the
    end-to-end proof (its ramp is planted after partitioning and does appear in D). This asserts
    the mechanism directly, so a regression names the cause rather than the symptom.
    """
    import inspect

    from fold_cp_ops._internal.gemm_sm90_epilogue import GemmSm90EpilogueMixin
    from fold_cp_ops._internal.gemm_sm90_mma import GemmSm90MmaMixin

    assert "flat_divide" in inspect.getsource(GemmSm90EpilogueMixin.epi_retile_acc), (
        "epi_retile_acc must RE-LAY-OUT (flat_divide) rather than copy, or hooks that write acc "
        "after partitioning are discarded"
    )
    # The order lives in ``run_epilogue``, the extract-method the role delegates the per-work-tile
    # store to. Reading it there rather than in the role is the point: a derivation that dispatches
    # between two epilogues overrides ``run_epilogue``, so that is where the ordering has to hold.
    role = inspect.getsource(GemmSm90MmaMixin.run_epilogue)
    assert role.index("self.epilogue_partition(") < role.index("self.epi_visit_acc("), (
        "partitioning must precede epi_visit_acc, which is what makes the aliasing load-bearing"
    )


_PROBE_COMPILED = {}


def _run_probe(cls, A, B, D, tile_M, tile_N):
    """Compile (once per class+config) and launch a bare ``GemmSm90`` subclass.

    Args:
        cls: The subclass to launch; must need no epilogue arguments (``()`` is passed).
        A: A operand ``(1, M, K)``, k-major.
        B: B operand ``(1, N, K)``, k-major.
        D: fp32 output, **written in place**.
        tile_M: CTA tile M.
        tile_N: CTA tile N.

    Returns:
        None.
    """
    a_dt, d_dt = torch2cute_dtype_map[A.dtype], torch2cute_dtype_map[D.dtype]
    A_p, B_p, D_p, _ = perm3d(A, B, D, None)
    key = (cls.__name__, a_dt, d_dt, tile_M, tile_N)
    if key not in _PROBE_COMPILED:
        mA, mB, mD, mC, _, _, _, l = make_fake_gemm_tensors(
            a_dt, a_dt, d_dt, None, "k", "k", "n", None
        )
        _PROBE_COMPILED[key] = compile_gemm_kernel(
            cls,
            a_dt,
            (tile_M, tile_N),
            (1, 1, 1),
            False,
            True,
            False,
            (9, 0),
            mA,
            mB,
            mD,
            mC,
            (),
            make_fake_scheduler_args(False, False, l),
        )
    clusters = get_max_active_clusters(1)
    _PROBE_COMPILED[key](A_p, B_p, D_p, None, (), make_scheduler_args(clusters, 8, None, None))
