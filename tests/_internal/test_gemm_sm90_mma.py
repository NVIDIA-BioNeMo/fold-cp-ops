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
"""Tests for ``fold_cp_ops._internal.gemm_sm90_mma`` -- the compute layer.

The third of the three layer tests. ``test_gemm_sm90_load.py`` isolates the transport INTO the
accumulator and ``test_gemm_sm90_epilogue.py`` the transport OUT of it; this one covers what
happens in between.

**It is shaped differently from the other two, and the reason is worth stating.** Those two isolate
a *transport*, so a probe can substitute known data at one end and read it at the other. The MMA is
not a transport -- there is no way to hand it operands without going through the load, and no way
to read its accumulator without going through the epilogue. Substituting either end would test the
substitute.

So this file tests the MMA layer's two contracts that the assembled kernel genuinely cannot pin
down on its own:

1. **Compile-time: the atom's geometry.** ``_make_tiled_mma`` picks the WGMMA atom, whose K extent
   becomes the CTA tile's K -- the k-tile size the *loader* stages against. A wrong value here does
   not produce a wrong number: the loader and the consumer disagree about how much data a stage
   holds, which is a hang or a torn tile. Host-side, no GPU.

2. **Runtime: the accumulator is zeroed per work tile.** ``mma`` zero-initialises on k-tile 0 and
   accumulates thereafter. When a persistent CTA takes SEVERAL work tiles, a missing re-zero adds
   the previous tile's result into this one. The end-to-end tests mostly run grids where each CTA
   takes one tile, so they would not see it; this forces the many-tiles-per-CTA case and checks the
   result integer-exactly.

Contract 2 is the one that catches a real defect class, and it is checked with
``assert_gemm_exact``: with integer operands the sum is exact, so a leaked accumulator shifts the
answer by a whole integer rather than by something a tolerance might swallow.
"""

import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from fold_cp_ops._internal.gemm_sm90_mma import NamedBarrierGemm
from fold_cp_ops.kernels.gemm_sm90 import GemmSm90
from fold_cp_ops.testing.numerics import assert_gemm_exact, integer_operands, max_exact_operand

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(_SM != 9, reason=f"needs sm_90; this GPU is sm_{_SM}0")


@pytest.fixture
def mlir_ctx():
    """An MLIR Context, Module and InsertionPoint -- what ``_make_tiled_mma`` needs to emit into.

    Building a WGMMA atom is *metaprogramming*: it emits layout and type algebra that the compiler
    constant-folds away, but it still has to emit it somewhere. All three pieces are load-bearing --
    the builder interns its types in the **Context**, the ops it creates are owned by the
    **Module**, and the **InsertionPoint** says where they go. Supplying only the Context leaves the
    ops with nowhere to land, which manifests as heap corruption on a later call rather than an
    exception (measured and documented in ``tests/_internal/compile_time/conftest.py``).

    Production never needs this: ``@cute.jit`` establishes all three for the duration of a trace,
    which is why ``_make_tiled_mma`` carries no decorator and is still safe at its real call site.

    Duplicated from the ``compile_time`` conftest rather than hoisted to a shared one: that fixture
    is function-scoped precisely so an ambient context never overlaps a GPU test, and moving it up a
    directory would put it in scope for every ``tests/_internal`` module including the ones that
    launch kernels.

    Yields:
        None. Used purely for its enter/exit side effect.
    """
    from cutlass._mlir import ir

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield


def _configured(tile_M, tile_N, a_dtype=cutlass.BFloat16, **kwargs):
    """Build a ``GemmSm90`` and its WGMMA atom, without a launch.

    ``_make_tiled_mma`` takes the operand facts as ARGUMENTS, which normally arrive from the tensors
    in ``__call__``. Passing them directly is what lets the atom's geometry be checked without
    compiling anything -- and it is the reason they are arguments rather than read off ``self``: the
    atom is built BEFORE the call parameters are bound, because the CTA tile's K is read back off
    it.

    Args:
        tile_M: CTA tile M.
        tile_N: CTA tile N.
        a_dtype: Operand type for both A and B.
        **kwargs: Forwarded to ``GemmSm90.__init__`` (``pingpong``, ...).

    Returns:
        ``(instance, tiled_mma, cta_tile_k)`` -- the functor, the atom it built, and the k-tile size
        the atom publishes. The atom is NOT on the instance: it carries MLIR values, so it is
        returned and passed to ``kernel()`` as an argument.
    """
    from cutlass.utils import LayoutEnum

    g = GemmSm90(Float32, a_dtype, (tile_M, tile_N), (1, 1, 1), **kwargs)
    k_major = LayoutEnum.ROW_MAJOR
    tiled_mma = g._make_tiled_mma(a_dtype, k_major, k_major)
    cta_tile_k = cute.size(tiled_mma.shape_mnk, mode=[2]) * GemmSm90._MMA_INST_TILE_K
    return g, tiled_mma, cta_tile_k


# ── contract 1: the atom's geometry, and the k-tile size it publishes ─────────────────────────
@pytest.mark.parametrize(
    "tile_M,tile_N,pingpong",
    [
        (64, 128, False),
        (128, 128, False),
        (128, 256, False),
        (192, 128, False),  # atom_layout (3, 1): three warpgroups down M
        (192, 256, False),  # atom_layout (1, 2): N is SPLIT across two warpgroups
        (256, 128, False),
        (320, 128, False),  # atom_layout (1, 2): tile_M / 64 is odd, so N splits
        (64, 128, True),
        (128, 192, True),
    ],
)
def test_setup_tiled_mma_builds_an_atom_that_covers_the_cta_tile(
    mlir_ctx, tile_M, tile_N, pingpong
):
    """The atom's N-mode times the N-split equals ``tile_N``, and its M-mode tiles ``tile_M``.

    The WGMMA atom is built with ``tiler_mn=(64, tile_N // atom_layout_n)``. If that N does not
    multiply back up to ``tile_N``, part of the tile is never computed -- and because the epilogue
    partitions against the accumulator it was given, the missing part is simply absent rather than
    wrong, which is far harder to see in an output.

    The 192 and 320 rows are the ones that matter: those are the ``tile_M`` values where the split
    goes down N instead of M, because ``tile_M / 64`` is not an even count of warpgroups.
    """
    g, tiled_mma, _ = _configured(tile_M, tile_N, pingpong=pingpong)
    atom_m, atom_n, atom_k = g.atom_layout_mnk
    shape = tiled_mma.shape_mnk
    assert cute.size(shape, mode=[1]) * atom_n == tile_N, (
        f"atom N {cute.size(shape, mode=[1])} x split {atom_n} != tile_N {tile_N}"
    )
    assert tile_M % (64 * atom_m) == 0 or atom_m * 64 == tile_M, (
        f"atom M coverage {64 * atom_m} does not tile tile_M {tile_M}"
    )
    assert atom_k == 1, "K is never split across warpgroups on SM90"


@pytest.mark.parametrize("tile_M,tile_N", [(64, 128), (128, 128), (128, 256), (256, 128)])
@pytest.mark.parametrize("a_dtype", [cutlass.BFloat16, cutlass.Float16])
def test_setup_tiled_mma_publishes_the_k_tile_size_the_loader_stages_against(
    mlir_ctx, tile_M, tile_N, a_dtype
):
    """``cta_tile_k`` is ``mma_inst_k * _MMA_INST_TILE_K`` -- 64 for a 16-bit operand.

    **This is the value the LOAD layer tiles K by**, via ``_k_tile_cnt`` and the SMEM layouts. The
    two layers never negotiate it; it is read off the atom and bound as a call parameter, and the
    load layer reads ``cta_tile_shape_mnk[2]``. So a change here silently changes how much data a
    pipeline stage holds, and the symptom is a hang or a torn tile rather than a wrong number --
    which is why it is asserted directly instead of being left to an end-to-end test.

    Before the call parameters are bound there is no ``cta_tile_shape_mnk`` at all; that is asserted
    too, because a placeholder K is precisely the thing that used to make "it was already 64" pass
    this by accident.
    """
    g = GemmSm90(Float32, a_dtype, (tile_M, tile_N), (1, 1, 1))
    with pytest.raises(AttributeError):
        g.cta_tile_shape_mnk  # noqa: B018 -- no operands bound, so there is no atom and no K
    g, tiled_mma, cta_tile_k = _configured(tile_M, tile_N, a_dtype=a_dtype)
    inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
    assert cta_tile_k == inst_k * 4
    assert cta_tile_k == 64, f"a 16-bit operand gives a 64-wide k-tile; got {cta_tile_k}"
    assert g.tile_shape_mn == (tile_M, tile_N), "M and N must be left alone"


def test_the_named_barriers_are_distinct_and_leave_barrier_zero_free():
    """Every named barrier has its own id, and none is 0.

    Hardware named barriers are a scarce, globally numbered per-CTA resource, and **barrier 0 is
    reserved for ``__syncthreads``**. Two roles sharing an id is a silent cross-role sync: each
    waits for threads it does not expect, which is a hang under load and a torn buffer otherwise.
    Neither is diagnosable from an output, so the ids are asserted here rather than trusted to
    ``enum.auto()`` staying as written.
    """
    ids = [int(b) for b in NamedBarrierGemm]
    assert len(ids) == len(set(ids)), f"duplicate barrier ids: {ids}"
    assert 0 not in ids, "barrier 0 is reserved for __syncthreads()"
    assert min(ids) == 1
    # The ping-pong pair is addressed as `base + warp_group_idx`, so the two ids must be adjacent
    # and the second must not collide with the next enum member.
    assert int(NamedBarrierGemm.MmaWG1) == int(NamedBarrierGemm.MmaWG0) + 1
    assert int(NamedBarrierGemm.EpiWG1) == int(NamedBarrierGemm.EpiWG0) + 1


def test_the_register_split_leaves_the_load_warpgroup_enough_to_issue_tma():
    """The MMA/load register split is asymmetric but never starves either side.

    The load warpgroup gets a small budget because it only issues descriptors; the MMA warpgroups
    get the rest because the accumulator fragment is this kernel's register pressure. Both budgets
    must be positive multiples of 8 (Hopper's allocation granularity) and must fit the file.
    """
    for tile in ((64, 128), (128, 128), (128, 256), (192, 128), (256, 128)):
        g = GemmSm90(Float32, cutlass.BFloat16, tile, (1, 1, 1))
        assert g.num_regs_load >= 24, f"tile={tile}: load budget {g.num_regs_load} is too small"
        assert g.num_regs_mma > g.num_regs_load, f"tile={tile}: the split is backwards"
        assert g.num_regs_load % 8 == 0 and g.num_regs_mma % 8 == 0
        mma_threads = g.mma_warp_groups * 128
        other = g.threads_per_cta - mma_threads
        assert mma_threads * g.num_regs_mma + other * g.num_regs_load <= 65536


# ── contract 2: the accumulator is re-zeroed for every work tile ──────────────────────────────
@requires_sm90
@pytest.mark.parametrize("tiles_per_cta", [2, 4, 9])
def test_the_accumulator_does_not_leak_between_work_tiles(tiles_per_cta):
    """A persistent CTA that takes several work tiles must not accumulate across them.

    ``mma`` zero-initialises the accumulator on k-tile 0 of each work tile and accumulates after.
    Drop the re-zero and tile *n* comes out as the sum of tiles 0..n -- which is a **whole-integer**
    error under integer operands, so ``assert_gemm_exact`` catches it at the first leaked element
    rather than as a tolerance breach.

    The end-to-end grid usually hands each CTA one tile, so it would not exercise this at all. Here
    the problem is deliberately sized to more tiles than the GPU has SMs' worth of work, forcing
    reuse: ``tiles_per_cta`` work tiles per resident CTA.

    Runs with ``persistent=True`` -- the whole point is the persistent loop; a one-CTA-per-tile grid
    gives every tile a fresh accumulator and cannot fail this no matter what ``mma`` does.
    """
    from fold_cp_ops.kernels.gemm import gemm

    tile = 128
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    n_tiles = n_sm * tiles_per_cta
    M, N = tile, tile * n_tiles  # one row of work tiles, n_tiles of them
    K = 128
    P = max_exact_operand(K, torch.float16)

    g = torch.Generator(device="cuda").manual_seed(31)
    A = integer_operands((1, M, K), torch.float16, "cuda", generator=g, bound=P)
    B = integer_operands((1, N, K), torch.float16, "cuda", generator=g, bound=P)
    D = torch.zeros(1, M, N, device="cuda", dtype=torch.float32)

    gemm(A, B, D, None, None, tile, tile, 1, 1, persistent=True)

    assert_gemm_exact(D, A, B, what=f"{n_tiles} work tiles over {n_sm} SMs")


@requires_sm90
def test_a_persistent_and_a_one_shot_launch_agree_bit_for_bit():
    """The persistent loop and one-CTA-per-tile produce identical bytes.

    They differ only in *who* computes each tile, and with integer operands each tile's value is
    exact, so the two must agree exactly. A difference means the persistent path carries state
    across tiles that the one-shot path does not -- the accumulator being the obvious candidate,
    but also the epilogue's subtile counter and the pipeline states.

    This is the same property as the test above approached from the other side: that one compares
    against an independent reference, this one against the kernel's own simpler schedule, so a
    defect present in BOTH the reference and the kernel could not hide in either.
    """
    from fold_cp_ops.kernels.gemm import gemm

    M, N, K = 512, 1024, 128
    P = max_exact_operand(K, torch.float16)
    g = torch.Generator(device="cuda").manual_seed(32)
    A = integer_operands((1, M, K), torch.float16, "cuda", generator=g, bound=P)
    B = integer_operands((1, N, K), torch.float16, "cuda", generator=g, bound=P)

    out = {}
    for persistent in (True, False):
        D = torch.zeros(1, M, N, device="cuda", dtype=torch.float32)
        gemm(A, B, D, None, None, 128, 128, 1, 1, persistent=persistent)
        out[persistent] = D

    assert torch.equal(out[True], out[False]), (
        f"persistent and one-shot disagree on "
        f"{int((out[True] != out[False]).sum())}/{out[True].numel()} elements"
    )
    assert_gemm_exact(out[True], A, B, what="persistent")


# ── the two extracted seams: MmaFragments, mma_setup_fragments, mma_consume_work_tile ─────────
def test_mma_fragments_binds_by_keyword_and_refuses_a_typo():
    """The bundle is keyword-only and slot-checked, so a misspelled field fails at trace time.

    ``acc`` and ``acc_slow`` are the same shape and dtype, so a positional constructor would let
    them be swapped with no diagnostic -- the fp8 slow-accumulate path would accumulate into the
    wrong register set and produce plausible-looking numbers. ``__slots__`` turns a typo into an
    ``AttributeError`` where it happens rather than a ``None`` read several frames later.
    """
    from fold_cp_ops._internal.gemm_sm90_mma import MmaFragments

    f = MmaFragments(thr_mma=1, acc=2, acc_slow=None, tCrA=3, tCrB=4, mma_fn=5)
    assert f.acc == 2 and f.extra is None, "extra must default to None on the base"
    with pytest.raises(TypeError, match=r"missing field"):
        MmaFragments(acc=2)
    with pytest.raises(AttributeError):
        MmaFragments(thr_mma=1, acc=2, acc_slow=None, tCrA=3, tCrB=4, mma_fn=5, accum=9)


def test_the_mma_seams_are_plain_defs_not_jit_wrapped():
    """``mma_setup_fragments`` and ``mma_consume_work_tile`` must NOT be ``@cute.jit``.

    This is a structural constraint, not a style choice. ``@cute.jit`` flattens every argument into
    MLIR values and cannot accept or return a Python object, so a jit-wrapped seam could not take
    or return :class:`MmaFragments` at all -- it would fail with ``DSLTreeFlattenError`` the first
    time a derivation used it. Both are therefore plain ``def``, which is safe only because every
    branch inside them is ``const_expr``.

    Guarded here because "make it jit like the others" is a natural-looking edit that breaks the
    extension point for everyone downstream while the in-tree tests stay green.
    """
    for name in ("mma_setup_fragments", "mma_consume_work_tile"):
        fn = getattr(GemmSm90, name)
        assert not hasattr(fn, "__wrapped__"), (
            f"{name} is decorated; a @cute.jit seam cannot carry an MmaFragments bundle"
        )


def test_the_role_builds_fragments_only_through_the_seam():
    """``mma_warpgroup_role`` must not re-inline the fragment construction it delegates.

    Read from source. If someone hoists ``partition_fragment_ABC`` back into the role "for
    clarity", every derivation's override stops being called and the in-tree tests still pass --
    the base's own answer is unchanged. Only a source check catches that.
    """
    import inspect

    from fold_cp_ops._internal.gemm_sm90_mma import GemmSm90MmaMixin

    src = inspect.getsource(GemmSm90MmaMixin.mma_warpgroup_role)
    assert "self.mma_setup_fragments(" in src
    assert "self.mma_consume_work_tile(" in src
    assert "partition_fragment_ABC" not in src, (
        "the role must build fragments through the seam, or overrides are dead code"
    )
    assert "self.mma(" not in src, "the per-tile MMA must go through mma_consume_work_tile"


@requires_sm90
def test_an_enriched_fragment_bundle_flows_through_unchanged():
    """A derivation may add state to the bundle without perturbing the answer.

    The base role must read only ``acc`` / ``acc_slow`` / ``mma_fn`` off the bundle and never reach
    around it. A subclass that overrides ``mma_setup_fragments``, calls ``super()``, and attaches
    its own payload to ``extra`` -- which is what the fused-LayerNorm kernel does with its
    statistics copy -- must therefore produce **bit-identical** output.

    That is the actual plug-and-play claim, tested rather than asserted.
    """
    from fold_cp_ops.kernels.gemm import gemm
    from fold_cp_ops.kernels.gemm_sm90 import GemmSm90 as _Base

    class _Enriched(_Base):
        """Attaches a payload to ``extra``; changes nothing the base reads."""

        def mma_setup_fragments(self, tiled_mma, sA, sB, wg_layout, wg_idx, mma_ctx=None):
            """Delegate, then attach an opaque payload. Returns the same bundle otherwise."""
            frags = super().mma_setup_fragments(tiled_mma, sA, sB, wg_layout, wg_idx, mma_ctx)
            frags.extra = ("a stand-in for the LN statistics state", frags.tCrA)
            return frags

    M, N, K = 512, 512, 256
    P = max_exact_operand(K, torch.float16)
    g = torch.Generator(device="cuda").manual_seed(41)
    A = integer_operands((1, M, K), torch.float16, "cuda", generator=g, bound=P)
    B = integer_operands((1, N, K), torch.float16, "cuda", generator=g, bound=P)

    base_out = torch.zeros(1, M, N, device="cuda", dtype=torch.float32)
    gemm(A, B, base_out, None, None, 128, 128, 1, 1)
    assert_gemm_exact(base_out, A, B, what="base")

    enriched = torch.zeros(1, M, N, device="cuda", dtype=torch.float32)
    _run_with(_Enriched, A, B, enriched, 128, 128)
    assert torch.equal(base_out, enriched), (
        f"enriching MmaFragments.extra changed "
        f"{int((base_out != enriched).sum())}/{base_out.numel()} elements"
    )


@requires_sm90
def test_the_mma_carry_threads_across_work_tiles():
    """``mma_carry`` returned for tile *n* arrives as the argument for tile *n+1*, per CTA.

    This is the mechanism the fused-LayerNorm cluster path needs: it carries its reduction and
    empty-barrier phases from one work tile to the next, and on ``main`` the only way to hold
    loop-carried state like that was to own the whole work-tile loop -- i.e. to fork ``kernel()``.

    Made observable by a probe that counts tiles in the carry and writes the count into the
    accumulator, so each 128x128 output block reports how many tiles its CTA had already done.
    The assertions are scheduler-independent on purpose:

    * every block is CONSTANT -- the carry is uniform within a work tile;
    * the minimum is 1 -- each CTA's first tile sees the seed from ``mma_initial_carry``;
    * the maximum is >= 1 -- **the carry actually advanced**. If it were dropped (the base's
      ``None`` returned unconditionally) every block would read 0 and this is the assertion that
      fails.
    """
    from fold_cp_ops.kernels.gemm_sm90 import GemmSm90 as _Base

    class _Counting(_Base):
        """Counts work tiles in the carry and stamps the count into the accumulator."""

        def mma_initial_carry(self):
            """Seed the carry as Int32, not None: the DSL forbids a type change across the loop."""
            return Int32(0)

        def mma_consume_work_tile(
            self, frags, ab_pipeline, ab_read_state, len_k, warp_group_idx, mma_carry=None
        ):
            """Run the real MMA, then overwrite acc with this CTA's tile counter."""
            ab_read_state, _ = super().mma_consume_work_tile(
                frags, ab_pipeline, ab_read_state, len_k, warp_group_idx, None
            )
            count = mma_carry + Int32(1)
            # PLAIN `range`, not `cutlass.range_constexpr`. The seam is a plain `def` -- it has to
            # be, to carry a Python bundle across the call -- so it does NOT get the DSL
            # preprocessor, and range_constexpr raises "should be preprocessed by preprocessor".
            # A compile-time-constant trip count unrolls under plain `range` at trace time anyway,
            # which is all range_constexpr does. `const_expr` IS fine here: it is an ordinary
            # function that returns its argument when unpreprocessed.
            for i in range(cute.size(frags.acc)):
                frags.acc[i] = Float32(count)
            return ab_read_state, count

    tile = 128
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    n_tiles = n_sm * 3
    M, N, K = tile, tile * n_tiles, 128
    g = torch.Generator(device="cuda").manual_seed(42)
    A = integer_operands((1, M, K), torch.float16, "cuda", generator=g, bound=4)
    B = integer_operands((1, N, K), torch.float16, "cuda", generator=g, bound=4)
    D = torch.zeros(1, M, N, device="cuda", dtype=torch.float32)
    _run_with(_Counting, A, B, D, tile, tile, persistent=True)

    blocks = D[0].reshape(M, n_tiles, tile).permute(1, 0, 2)  # (n_tiles, M, tile)
    lo = blocks.amin(dim=(1, 2))
    hi = blocks.amax(dim=(1, 2))
    assert torch.equal(lo, hi), (
        f"{int((lo != hi).sum())} work tiles have a non-constant counter -- the carry is not "
        f"uniform within a tile"
    )
    assert float(lo.min()) == 1.0, "each CTA's first tile must see mma_initial_carry()'s seed"
    assert float(hi.max()) >= 2.0, (
        "the carry never advanced: every work tile reported the seed, so the value returned for "
        "tile n did not arrive as the argument for tile n+1"
    )


_RUN_COMPILED = {}


def _run_with(cls, A, B, D, tile_M, tile_N, persistent=True):
    """Compile (once per class+config) and launch a bare ``GemmSm90`` subclass over real operands.

    A local twin of ``tests/kernels/test_gemm_sm90.py``'s ``run_base_gemm``, keyed additionally on
    the CLASS so two probes in this module do not share one compiled artifact.

    Args:
        cls: The ``GemmSm90`` subclass to launch. Must need no epilogue arguments -- ``()`` is
            passed, i.e. the base class's own default hooks and a plain ``D = A @ B^T`` store.
        A: A operand, ``(1, M, K)``, k-major.
        B: B operand, ``(1, N, K)``, k-major.
        D: Output, **written in place**. fp32 so integer results survive exactly.
        tile_M: CTA tile M.
        tile_N: CTA tile N.
        persistent: Whether to launch a persistent grid. Required True for anything that depends
            on one CTA taking several work tiles.

    Returns:
        None.
    """
    from fold_cp_ops._internal.arch import get_max_active_clusters
    from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
    from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
        compile_gemm_kernel,
        make_fake_gemm_tensors,
        make_fake_scheduler_args,
        make_scheduler_args,
        perm3d,
    )

    a_dt, d_dt = torch2cute_dtype_map[A.dtype], torch2cute_dtype_map[D.dtype]
    A_p, B_p, D_p, _ = perm3d(A, B, D, None)
    key = (cls.__name__, a_dt, d_dt, tile_M, tile_N, persistent)
    if key not in _RUN_COMPILED:
        mA, mB, mD, mC, _, _, _, l = make_fake_gemm_tensors(
            a_dt, a_dt, d_dt, None, "k", "k", "n", None
        )
        _RUN_COMPILED[key] = compile_gemm_kernel(
            cls,
            a_dt,
            (tile_M, tile_N),
            (1, 1, 1),
            False,
            persistent,
            False,
            (9, 0),
            mA,
            mB,
            mD,
            mC,
            (),
            make_fake_scheduler_args(False, False, l),
        )
    clusters = get_max_active_clusters(1) if persistent else 0
    _RUN_COMPILED[key](A_p, B_p, D_p, None, (), make_scheduler_args(clusters, 8, None, None))


# ── the two seams the fused-LayerNorm mainloop needs: MmaContext and the per-k-tile hook ──────


def test_mma_context_binds_by_keyword_and_refuses_a_typo():
    """Like `MmaFragments`: keyword-only and slot-checked, so a typo fails where it is written.

    Two of its four fields are opaque containers -- ``storage`` and ``epi_smem_tensors`` -- and
    swapping them would produce a kernel that indexes one as the other. ``__slots__`` and a missing
    -field check are what make that impossible rather than merely unlikely.
    """
    from fold_cp_ops._internal.gemm_sm90_mma import MmaContext

    c = MmaContext(storage=1, epi_smem_tensors=(2,), epilogue_params=3, tidx=4)
    assert c.tidx == 4 and c.epi_smem_tensors == (2,)
    with pytest.raises(TypeError, match=r"missing field"):
        MmaContext(storage=1)
    with pytest.raises(AttributeError):
        MmaContext(storage=1, epi_smem_tensors=(2,), epilogue_params=3, tidx=4, tid=5)


def test_the_role_hands_the_context_to_the_fragment_seam():
    """The context is BUILT in the role and passed down; a derivation cannot reach it otherwise.

    The statistics scratch is allocated by an EPILOGUE op and ``eps`` arrives in the EPILOGUE
    arguments, so a fused mainloop has no route to either except this one. Read from source, because
    a change that stopped passing it would leave the base's own answer unchanged and every fused
    kernel silently unable to find its scratch.
    """
    import inspect

    from fold_cp_ops._internal.gemm_sm90_mma import GemmSm90MmaMixin

    src = inspect.getsource(GemmSm90MmaMixin.mma_warpgroup_role)
    assert "MmaContext(" in src
    assert "epi_smem_tensors=epi_smem_tensors" in src
    assert "epilogue_params=epilogue_params" in src
    # tidx must be the value already reduced modulo the warpgroup size under ping-pong, or two
    # warpgroups would claim the same rows of a per-warpgroup buffer.
    assert src.index("tidx = tidx % self.num_threads_per_warp_group") < src.index("MmaContext(")


def test_the_per_k_tile_hook_is_called_once_per_tile_after_the_mma():
    """``on_ktile`` fires for EVERY k-tile -- prologue and mainloop -- and after the WGMMA is issued.

    Both halves matter. Missing the prologue tile would silently drop the first k-tile's
    contribution from the statistics, which at large K is a small enough error to sit inside any
    tolerance. Firing BEFORE the WGMMA would serialize the extra shared-memory read against the MMA
    warps' own operand fetch instead of overlapping the flying instruction.
    """
    import inspect

    from fold_cp_ops._internal.gemm_sm90_mma import GemmSm90MmaMixin

    src = inspect.getsource(GemmSm90MmaMixin.mma)
    body = src[src.index('"""', src.index('"""') + 3) :]
    # THREE, not two: the prologue loop once, and the mainloop TWICE -- it is written out for a
    # dynamic trip count and again for a static one, and the two bodies must stay identical. The
    # duplication is not avoidable by factoring the body into a local function; the DSL re-derives
    # `mma` from source, so a nested closure loses its free variables.
    assert body.count("on_ktile(ab_read_state.index)") == 4, (
        "once in EACH spelling of the prologue and once in EACH spelling of the mainloop -- the "
        "prologue is spelled twice for the same reason the mainloop is, and for one more: given a "
        "static k-tile count its look-ahead guard must compare against THAT count, or the guard "
        "stays dynamic and re-introduces the per-iteration branch the static count exists to remove"
    )
    for call in [i for i in range(len(body)) if body.startswith("on_ktile(", i)]:
        before = body[:call]
        assert before.rindex("mma_fn(") > before.rindex("consumer_wait("), (
            "the hook must run AFTER the WGMMA is issued, so its reads overlap it"
        )


def test_the_static_count_spelling_never_compares_against_the_runtime_count():
    """Given a static k-tile count, every loop bound AND guard must use THAT count.

    This is the whole point of passing one. A ``range_constexpr`` loop whose look-ahead guard reads
    the RUNTIME ``k_tile_cnt`` is still dynamic: the compiler cannot fold it, so each unrolled
    iteration keeps a compare and a conditional ``consumer_try_wait`` region -- measured at ~9 extra
    branches and ~37 extra instructions per k-tile against the upstream, which compares against its
    static count throughout. The mistake is invisible in the output (byte-identical) and invisible
    in the config (same tile, same stages), so a source guard is what catches it.

    Scans source rather than codegen because the distinction IS syntactic: ``k_tile_cnt`` versus
    ``k_tile_cnt_const`` in the comparison, and ``cutlass.range`` versus ``cutlass.range_constexpr``
    in the loop.
    """
    import inspect

    from fold_cp_ops._internal.gemm_sm90_mma import GemmSm90MmaMixin

    src = inspect.getsource(GemmSm90MmaMixin.mma)
    assert src.count("if k_tile + 1 < k_tile_cnt:") == 2, (
        "the RUNTIME guard belongs only in the two dynamic-count spellings (prologue + mainloop)"
    )
    assert src.count("if const_expr(k_tile + 1 < k_tile_cnt_const):") == 2, (
        "the two static-count spellings (prologue + mainloop) must fold their guard at compile "
        "time; comparing against the runtime count leaves the branch the static count removes"
    )
    static_half = src[src.index("if const_expr(k_tile_cnt_const is None):") :]
    assert "cutlass.range_constexpr(num_prologue_mma)" in static_half, (
        "with a static count the prologue and drain trip counts are Python ints and must unroll"
    )


def test_the_per_k_tile_hook_is_pruned_when_absent():
    """A plain GEMM emits nothing for it: the guard is ``const_expr``, not a runtime branch."""
    import inspect

    from fold_cp_ops._internal.gemm_sm90_mma import GemmSm90MmaMixin

    src = inspect.getsource(GemmSm90MmaMixin.mma)
    # Four guards: both spellings of the prologue plus both spellings of the mainloop.
    assert src.count("const_expr(on_ktile is not None)") == 4
    assert "on_ktile: Optional[Callable] = None" in src, "absent by default"


# ------------------------------------------------------------------------------------------------
# The wide-A2A producer seam.
#
# This parent carries one A2A hook that the kernel this ports from carries on the staged dual's OWN
# `kernel` method. That kernel does not exist here -- every fusion shares this mainloop -- so the
# call site is visible to every kernel rather than to one, and its inertness for all the others
# rests on an invariant rather than on structure. These tests are that invariant, written down.
# ------------------------------------------------------------------------------------------------

#: The flag whose presence turns the seam on. Written only by the front A2A subclass.
_WIDE_SEAM_FLAG = "_a2a_ib_wide"
#: The predicate the seam CALLS. It is defined only by the front A2A subclass -- deliberately, since
#: a default on this parent to make a subclass safe would be a divergence from the ported kernel.
_WIDE_SEAM_PREDICATE = "_decoupled_active"
#: Retired code is MOVED here, not deleted, so every tree walk must skip it. A stale class answering
#: one of these scans would report a violation that no live code can reach, or hide a live one.
_TRASH_DIR = "trash_to_be_removed"


def _package_sources():
    """Every live ``.py`` under the ``fold_cp_ops`` package root, as (path, parsed tree).

    Input requirements: none; the root is derived from the module under test rather than from the
    CWD, so the scan is the same under any launcher. ``trash_to_be_removed/`` is skipped at any
    depth -- see :data:`_TRASH_DIR`.

    Returns: list of ``(pathlib.Path, ast.Module)``. Non-empty by construction; the callers assert
        it, because an empty scan makes every "no offenders" assertion below pass vacuously.
    """
    import ast as _ast
    import pathlib

    from fold_cp_ops._internal import gemm_sm90_mma as _mod

    root = pathlib.Path(_mod.__file__).parent.parent
    out = []
    for path in sorted(root.rglob("*.py")):
        if _TRASH_DIR in path.parts:
            continue
        out.append((path, _ast.parse(path.read_text())))
    return out


def _classes_writing(tree, attr):
    """Names of classes in `tree` that ASSIGN `attr`, as ``self.<attr> = …`` or a class-body default.

    Args:
        tree: A parsed ``ast.Module``. Passing a module rather than a path is what lets the positive
            control below feed a synthetic source and prove this function discriminates.
        attr: The attribute name, without the leading dot.

    Returns: sorted list of class names. A class assigning it in several methods appears once.
    """
    import ast as _ast

    found = set()
    for cls in [n for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)]:
        for node in _ast.walk(cls):
            if not isinstance(node, (_ast.Assign, _ast.AnnAssign, _ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, _ast.Assign) else [node.target]
            for t in targets:
                if (
                    isinstance(t, _ast.Attribute)
                    and t.attr == attr
                    and isinstance(t.value, _ast.Name)
                    and t.value.id == "self"
                ):
                    found.add(cls.name)
                elif isinstance(t, _ast.Name) and t.id == attr:
                    found.add(cls.name)
    return sorted(found)


def _classes_defining(tree, method):
    """Names of classes in `tree` that define a method called `method`. Same contract as above."""
    import ast as _ast

    return sorted(
        {
            cls.name
            for cls in [n for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)]
            if any(
                isinstance(f, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and f.name == method
                for f in cls.body
            )
        }
    )


def test_the_wide_producer_seam_guards_in_short_circuit_order():
    """The predicate the seam calls must be the LAST conjunct, behind ``getattr``-defaulted flags.

    This is the whole safety argument for putting the seam on the SHARED mainloop. Every conjunct
    before ``_decoupled_active()`` is a ``getattr(self, …, False)``, which cannot raise on a class
    that has never heard of the flag; Python's ``and`` then short-circuits and the call is never
    reached. Reorder the conjuncts -- put the call first, or "simplify" a ``getattr`` to a plain
    attribute read -- and every kernel sharing this mainloop raises ``AttributeError`` at trace.

    The failure would not be subtle, but it would be introduced by someone tidying a boolean, which
    is exactly the edit nobody re-tests. Parsed rather than string-matched so reformatting the
    condition does not break the test and rewriting it does.
    """
    import ast as _ast
    import inspect
    import textwrap

    from fold_cp_ops._internal.gemm_sm90_mma import GemmSm90MmaMixin

    src = textwrap.dedent(inspect.getsource(GemmSm90MmaMixin.mma_warpgroup_role))
    tree = _ast.parse(src)
    guards = [
        n
        for n in _ast.walk(tree)
        if isinstance(n, _ast.If)
        and any(
            isinstance(c, _ast.Call)
            and isinstance(c.func, _ast.Attribute)
            and c.func.attr == "_a2a_wide_producer_finalize"
            for c in _ast.walk(n)
        )
    ]
    assert guards, (
        "the wide-A2A producer seam is GONE from mma_warpgroup_role. It is the one single-device "
        "hook the front A2A kernel needs from this parent; without it the CTA's trailing wide batch "
        "is never flushed and the consumer's count parity never closes."
    )
    guard = min(guards, key=lambda n: len(list(_ast.walk(n))))  # the innermost, i.e. the seam's own
    cond = guard.test
    assert isinstance(cond, _ast.Call) and getattr(cond.func, "id", None) == "const_expr", (
        "the seam's guard must be const_expr so the branch is pruned at trace rather than becoming "
        "a runtime branch in every kernel that shares this mainloop"
    )
    conjuncts = cond.args[0]
    assert isinstance(conjuncts, _ast.BoolOp) and isinstance(conjuncts.op, _ast.And)
    calls = [
        i
        for i, v in enumerate(conjuncts.values)
        if isinstance(v, _ast.Call)
        and isinstance(v.func, _ast.Attribute)
        and v.func.attr == _WIDE_SEAM_PREDICATE
    ]
    assert calls == [len(conjuncts.values) - 1], (
        f"{_WIDE_SEAM_PREDICATE}() must be the LAST conjunct (found at {calls} of "
        f"{len(conjuncts.values)}): every earlier one is a getattr-with-default that cannot raise, "
        "and the short circuit is what lets classes without the predicate share this mainloop."
    )
    for v in conjuncts.values[:-1]:
        assert (
            isinstance(v, _ast.Call)
            and getattr(v.func, "id", None) == "getattr"
            and len(v.args) == 3
        ), (
            "every conjunct before the predicate must be getattr(self, name, default) -- a bare "
            "self.<flag> raises AttributeError on every kernel that is not the front A2A one"
        )


def test_every_writer_of_the_wide_flag_also_defines_the_predicate_the_seam_calls():
    """A class that turns the seam ON must supply the predicate the seam then calls.

    This is the invariant the short-circuit order relies on, stated positively: the flag and the
    predicate travel together. Today both sets are EMPTY -- the A2A kernels are not ported yet -- so
    the subset holds trivially, and the sibling positive-control test is what proves the check can
    fail at all. It starts biting the moment the front A2A kernel lands, which is the point: the
    alternative was a grep somebody ran once.
    """
    sources = _package_sources()
    assert sources, "the package scan found no sources; every assertion below would pass vacuously"
    writers, definers = {}, set()
    for path, tree in sources:
        for c in _classes_writing(tree, _WIDE_SEAM_FLAG):
            writers[c] = path
        definers.update(_classes_defining(tree, _WIDE_SEAM_PREDICATE))
    orphans = {c: p for c, p in writers.items() if c not in definers}
    assert not orphans, (
        "these classes set "
        + _WIDE_SEAM_FLAG
        + " without defining "
        + _WIDE_SEAM_PREDICATE
        + ": "
        + ", ".join(f"{c} ({p.name})" for c, p in sorted(orphans.items()))
        + ". The shared mainloop's seam calls that predicate once the flag is on, so the class "
        "would raise AttributeError at trace. Define it on the subclass -- NOT as a default on the "
        "parent, which would be a divergence from the kernel this ports from."
    )


def test_the_writer_without_predicate_check_rejects_a_synthetic_violation():
    """Positive control: the check above must FAIL on a class that sets the flag and omits the rest.

    Without this the sibling test is unfalsifiable while the A2A kernels are unported -- it scans
    two empty sets and passes. A probe that cannot fail proves nothing, so the discrimination is
    demonstrated on a synthetic source instead of asserted.
    """
    import ast as _ast

    bad = _ast.parse(
        f"class Offender:\n    def configure(self):\n        self.{_WIDE_SEAM_FLAG} = True\n"
    )
    good = _ast.parse(
        "class Compliant:\n"
        "    def configure(self):\n"
        f"        self.{_WIDE_SEAM_FLAG} = True\n"
        f"    def {_WIDE_SEAM_PREDICATE}(self):\n"
        "        return True\n"
    )
    assert _classes_writing(bad, _WIDE_SEAM_FLAG) == ["Offender"], "the writer scan missed a writer"
    assert _classes_defining(bad, _WIDE_SEAM_PREDICATE) == [], "no predicate is defined here"
    assert _classes_writing(good, _WIDE_SEAM_FLAG) == ["Compliant"]
    assert _classes_defining(good, _WIDE_SEAM_PREDICATE) == ["Compliant"], (
        "the definer scan missed a definition; the subset check would then flag a compliant class"
    )


#: The ONLY classes entitled to write a gating flag, per flag. EXACT: an entry that no longer
#: matches fails just as loudly as an unlisted writer.
#:
#: **Why an allowlist replaced "nobody writes these".** The original assertion was "written
#: NOWHERE", which was true only while the A2A subclasses were unported and became false the moment
#: the front kernel landed. The invariant it stood for is narrower and is the one §5.1 states:
#: ``_a2a_ib_wide`` is written **only in the front kernel**. Asserting the stronger, false thing
#: does not make the check safer -- it makes it fire on correct code, and a check that fires on
#: correct code gets deleted rather than fixed. That is how a landmine detector is disarmed.
#:
#: **Why the set must be EXACT rather than a floor.** A subset check ("no writer outside this list")
#: rots in the other direction: an entry whose class was renamed or deleted silently stops
#: constraining anything, and the list decays into decoration. Same reasoning as the numeric guard's
#: pinned exempt-module set -- an un-pinned allowlist checks one thing less with every stale entry.
_WIDE_SEAM_ENTITLED_WRITERS = {
    "_a2a_ib_wide": frozenset({"DualGatedGemmDistSm90"}),
    "_a2a_enabled": frozenset({"DualGatedGemmDistSm90", "GemmSm90A2A"}),
}
#: The subpackage the entitled writers must live in. This is the byte-neutrality half of the claim:
#: the flags may be written, but only from the distributed half, so every SINGLE-DEVICE kernel still
#: const_expr-prunes the seam and its cubin cannot move.
_A2A_SUBPACKAGE = "distributed"


def test_the_wide_seam_is_inert_outside_the_a2a_subclasses():
    """Only the A2A subclasses set the gating flags, so the seam is pruned in every LOCAL kernel.

    This is the byte-neutrality claim as a test rather than an argument. The seam was added to a
    mainloop every fusion shares; it costs nothing while both flags are absent, and they are absent
    from the single-device half because the A2A subclasses live in ``distributed/``. If a future
    edit sets one of them on a local kernel, the branch stops being pruned and the parent's cubin
    moves -- precisely the thing the bring-back forbids, and precisely the edit whose author would
    not think to re-measure a cubin.

    Two independent assertions, because either alone is escapable:

    1. **The writer set is EXACT** (:data:`_WIDE_SEAM_ENTITLED_WRITERS`). A new class writing a flag
       fails -- including one under ``distributed/``, which is the case most worth catching, since
       that is where such an edit would naturally be made. A stale entry fails too, so the list
       cannot decay into decoration.
    2. **Every entitled writer lives under** ``distributed/``. This is what keeps the test about
       cubin byte-identity rather than about class naming: moving an A2A class into ``kernels/``
       would satisfy (1) and still put a flag-writer in the single-device half.
    """
    sources = _package_sources()
    assert sources, "the package scan found no sources; this assertion would pass vacuously"
    observed = {flag: {} for flag in _WIDE_SEAM_ENTITLED_WRITERS}
    for path, tree in sources:
        for flag in _WIDE_SEAM_ENTITLED_WRITERS:
            for c in _classes_writing(tree, flag):
                observed[flag][c] = path
    for flag, entitled in _WIDE_SEAM_ENTITLED_WRITERS.items():
        got = set(observed[flag])
        unlisted = sorted(f"{c} ({observed[flag][c].name})" for c in got - entitled)
        assert not unlisted, (
            f"class(es) not entitled to write {flag} now do: {', '.join(unlisted)}. If this is a "
            "new A2A subclass, add it to _WIDE_SEAM_ENTITLED_WRITERS *and* re-prove the parent's "
            "cubin byte identity -- the seam stops being const_expr-pruned wherever the flag is on."
        )
        stale = sorted(entitled - got)
        assert not stale, (
            f"_WIDE_SEAM_ENTITLED_WRITERS lists {stale} as writing {flag}, but no class does any "
            "more. An allowlist entry that matches nothing constrains nothing; drop it, so the "
            "list keeps meaning what it says."
        )
    misplaced = sorted(
        f"{c} ({p})"
        for flag in observed
        for c, p in observed[flag].items()
        if _A2A_SUBPACKAGE not in p.parts
    )
    assert not misplaced, (
        f"these flag-writers live outside {_A2A_SUBPACKAGE}/: {', '.join(misplaced)}. The seam is "
        "inert in the single-device half only because every writer is in the distributed half; a "
        "writer in kernels/ moves that kernel's cubin."
    )


def test_the_seam_inertness_check_rejects_a_synthetic_offender():
    """Positive control: an unlisted class writing a flag must be REFUSED, at every flag.

    The sibling above passes today because the three real writers are exactly the entitled ones --
    a state indistinguishable, from the outside, from a check that cannot fail. So the
    discrimination is demonstrated on synthetic sources rather than assumed: a class name that is
    not in the allowlist must be reported for each flag the allowlist covers.
    """
    import ast as _ast

    for flag, entitled in _WIDE_SEAM_ENTITLED_WRITERS.items():
        bad = _ast.parse(
            f"class NotEntitled:\n    def configure(self):\n        self.{flag} = True\n"
        )
        found = set(_classes_writing(bad, flag))
        assert found == {"NotEntitled"}, f"the writer scan missed a writer of {flag}"
        assert found - entitled == {"NotEntitled"}, (
            f"a class outside {sorted(entitled)} wrote {flag} and the allowlist did not catch it"
        )
        ok = _ast.parse(
            f"class {sorted(entitled)[0]}:\n    def configure(self):\n        self.{flag} = True\n"
        )
        assert not (set(_classes_writing(ok, flag)) - entitled), (
            f"an ENTITLED writer of {flag} was reported as an offender; the check would fire on "
            "correct code, which is how this kind of test gets deleted instead of fixed"
        )


@pytest.mark.parametrize("flag", sorted(_WIDE_SEAM_ENTITLED_WRITERS))
@pytest.mark.parametrize(
    "where", ["kernels/_synthetic.py", "distributed/_synthetic.py"], ids=["local", "distributed"]
)
def test_the_seam_inertness_assertion_fires_on_an_injected_offender(monkeypatch, flag, where):
    """The ASSERTION fails, not merely the scanner -- with the offender injected into the scan.

    The sibling control proves ``_classes_writing`` discriminates. That is a weaker claim than the
    one that matters: a scanner can be perfect while the assertion built on it is unreachable. So
    this drives the real test function with a synthetic source spliced into ``_package_sources`` and
    requires it to RAISE.

    Swept over ``where`` because the two placements must both fail for different reasons, and a
    check that caught only one would leave the other silent: a writer under ``kernels/`` is the
    byte-identity hazard, and a writer under ``distributed/`` is the one an author would most
    plausibly add -- an unlisted A2A subclass. Testing only the former would let the latter through,
    which is exactly the case the allowlist exists for.
    """
    import ast as _ast
    import pathlib
    import sys

    real = _package_sources

    def fake():
        out = list(real())
        out.append(
            (
                pathlib.Path(where),
                _ast.parse(f"class NotEntitled:\n    def c(self):\n        self.{flag} = True\n"),
            )
        )
        return out

    monkeypatch.setattr(sys.modules[__name__], "_package_sources", fake)
    with pytest.raises(AssertionError, match="NotEntitled"):
        test_the_wide_seam_is_inert_outside_the_a2a_subclasses()
