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

"""PERF GATE: ``fold_cp_ops.kernels.layernorm_dual_gated_gemm.layernorm_dual_gated_gemm``.

**Do NOT lock the GPU clock for this gate**, and do not harvest its pins with one either. Locking
makes a single measurement far more repeatable but SHIFTS THE MEAN: an H100's free-running clock
follows the 700 W power cap and hence the workload, so a locked pin and a free-running measurement
sample two different distributions.

**Read this beside ``test_benchmark_perf_dual_gated_gemm.py``.** That gate times the same mainloop
and the same gated fold WITHOUT the LayerNorm prologue, at the same shapes. The difference between
the two files IS the fusion's cost, and it is the number this gate exists to hold: the prologue
streams A twice and drains each WGMMA before releasing its tile, so it is expected to cost
something, and what must not drift is HOW MUCH. A regression that shows here and not there is in
the prologue; one that shows in both is in the mainloop underneath it.

**The k-tile sweep is the axis this kernel adds.** ``blk_k`` sets the k-loop trip count, so it
changes both the arithmetic and the number of pipeline turns. It is swept at one shape so the
differences are the tile and nothing else -- and note that unlike ``chunk_g``, the cells are NOT
computing the same bits, so this sweep measures a family of related kernels rather than one kernel
under different transport.

**The fused output gate is timed against its own absence.** Its whole design claim is that an
output-gate work tile is an ordinary work tile of a wider operand, so its cost should be
proportional to the extra tiles and nothing else. The pair of cells is what makes that checkable.

Timing goes through ``bench_utils.benchmark_single(mode="device")`` per CLAUDE.md -- never a
hand-rolled ``perf_counter`` loop around an async launch, which measures host dispatch and can
report a bandwidth above the physical link.

**What the harvested numbers say** (H100 80GB, bf16, max of three consecutive runs -- a pin set at
a lucky low is a gate that fails on the unlucky run, which teaches people to re-run it rather than
to read it):

* The **fusion's cost** is the gap to ``test_benchmark_perf_dual_gated_gemm.py``'s matching cell.
  Measured directly on the device timeline rather than through these entries: +1.74 us at
  ``chunk_g=1`` and +1.98 us at ``chunk_g=16`` for 4096x256x128, against the same kernel with the
  prologue compiled out. The kernel this reproduces costs +1.71 and +1.97 at the same cells, so the
  prologue is neutral to within 2%.
* The **layout pair** costs 0.0175 ms at ``chunk_g=1`` against 0.0144 at 16 -- 1.22x, at a shape
  where the two compute bit-identical results. All of it is the per-call host weight interleave the
  element-interleave layout needs, which is why the front door recommends the block layout and why
  the interleave sits inside the timed region.
* The **K tile** is monotone: 0.0175 / 0.0187 / 0.0236 ms at 64 / 32 / 16. The default is the fast
  one at every width, which is what ``auto_blk_k``'s widest-divisor policy assumes.
* The **optional terms are free**: 0.01443 to 0.01472 ms across all four LayerNorm-bias x mask
  cells, a 2% spread. The prologue's bias add is one fp32 add per element inside a loop that was
  already loading the gain, and the mask is one multiply in the epilogue.
* The **output gate costs 0.0613 ms against 0.0159 without it** -- 3.9x for 1.5x the work, and the
  excess is NOT in the kernel. This entry builds the combined ``[dual | W3]`` weight on the HOST on
  every call (an allocation and two copies), exactly as the kernel it reproduces does. A caller that
  runs the gate in a loop should hoist that construction; the cell is pinned with it included
  because that is what the entry point as written actually costs.

Harvest new pins (prints medians and TFLOP/s, skips the assertion)::

    CPO_PERF_MEASURE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s \
        tests/perf/test_benchmark_perf_layernorm_dual_gated_gemm.py
"""

import os

import pytest
import torch

from benchmark.distributed import bench_utils as BU
from tests.perf.calibration import assert_cell, assert_host_dispatch
from tests.perf.pins import load as load_pins
from fold_cp_ops.kernels.layernorm_dual_gated_gemm import layernorm_dual_gated_gemm
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from tests.kernels.test_layernorm_dual_gated_gemm import LN_DUAL_GATED  # the ONE declaration

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(
    _SM != 9, reason=f"pins are SM90 (H100) numbers; this GPU is sm_{_SM}0"
)

_MEASURE = os.environ.get("CPO_PERF_MEASURE", "0") == "1"

_ROUNDS, _WARMUP, _ITERS = 30, 10, 10

#: This gate's pinned numbers, loaded from the sibling JSON of the SAME NAME. Enforced in conftest:
#: a timed perf module that does not expose ``PINS`` FAILS rather than skips, because a gate
#: inventing its own number format is exactly what that check exists to catch.
PINS = load_pins(__file__)


def _bench(key, fn, flops):
    """Time ``fn``, then either assert against its pin or print it for harvesting.

    Args:
        key: The cell's key into :data:`PINS`.
        fn: A zero-argument callable that launches the kernel once. Must not synchronize --
            ``benchmark_single`` owns the CUDA events and the L2 flush.
        flops: The cell's floating-point work, used only for the reported TFLOP/s.

    Returns:
        None.

    Raises:
        AssertionError: If the measured median exceeds the pin by more than its own band.
        pytest.skip.Exception: If the cell is unpinned.
    """
    measure = lambda: (  # noqa: E731
        BU.benchmark_single(
            fn, mode="device", rounds=_ROUNDS, warmup=_WARMUP, iters=_ITERS
        ).median_ms
    )
    assert_cell(
        measure,
        key,
        PINS,
        __file__,
        flops=flops,
        describe=lambda ms: f"({flops / (ms * 1e-3) / 1e12:6.1f} TFLOP/s)",
    )


def _operands(M, N, K, dtype=torch.bfloat16, N3=0, x_gate=False):
    """Build one cell's operands, pitch-padded so any N is legal.

    Args:
        M: Token extent.
        N: Output feature extent -- the pre-activation is twice this, EXCEPT on the ``x_gate``
            path, where the two projections have separate N-wide accumulators and no 2N tile.
        K: Contraction extent (the model's hidden width). Must meet the 16-byte pitch floor.
        dtype: Operand and output element type.
        N3: Output-gate width, or 0 for no output gate.
        x_gate: Whether to build the separate pre-normalized gate activation, which selects the
            two-A kernel. Mutually exclusive with `N3`.

    Returns:
        A dict of keyword arguments for the front door, including the preallocated outputs. The
        outputs' row pitch is padded up to the 16-byte floor, so an off-grid N is timed rather than
        refused.
    """
    e = 16 // dtype.itemsize
    pad = lambda n: (n + e - 1) // e * e  # noqa: E731
    ops = {
        "x": torch.randn(M, K, device="cuda", dtype=dtype),
        "norm_weight": torch.randn(K, device="cuda", dtype=torch.float32),
        "Wg": torch.randn(N, K, device="cuda", dtype=dtype) * 0.1,
        "Wp": torch.randn(N, K, device="cuda", dtype=dtype) * 0.1,
        "PostAct": torch.empty(M, pad(N), device="cuda", dtype=dtype)[:, :N],
    }
    if N3:
        ops["W3"] = torch.randn(N3, K, device="cuda", dtype=dtype) * 0.1
        ops["b3"] = torch.randn(N3, device="cuda", dtype=torch.float32)
        ops["PostAct3"] = torch.empty(M, pad(N3), device="cuda", dtype=dtype)[:, :N3]
    if x_gate:
        ops["x_gate"] = torch.randn(M, K, device="cuda", dtype=dtype)
    return ops


def _call(ops, tile_M=128, tile_N=256, **kw):
    """Bind one launch of the front door over `ops`.

    Args:
        ops: The dict from :func:`_operands`.
        tile_M: CTA tile M.
        tile_N: CTA tile N over the 2N pre-activation.
        **kw: Forwarded to the front door (``chunk_g``, ``blk_k``, biases, mask, ...).

    Returns:
        A zero-argument callable that launches once and does not synchronize.
    """
    return lambda: layernorm_dual_gated_gemm(
        ops["x"],
        ops["norm_weight"],
        ops["Wg"],
        ops["Wp"],
        ops["PostAct"],
        tile_M,
        tile_N,
        W3=ops.get("W3"),
        b3=ops.get("b3"),
        PostAct3=ops.get("PostAct3"),
        x_gate=ops.get("x_gate"),
        **kw,
    )


#: The A2A-fused TriMul workflow's OWN shapes, DERIVED from the kernel matrix rather than listed.
#:
#: In that workflow the front projection consumes a local token shard of ``N_token x N_token/cp``
#: rows, so ``M = N_token**2 / cp``, and both projections are ``(D, D)`` so ``K == N == D``. That
#: correlation is why this is a list of CELLS and not an axis product -- but the RANGE it spans is
#: the matrix's, read off the ``workflow_M`` and ``workflow_D`` facets. Two things follow, and both
#: are the point of doing it this way:
#:
#: * a token count or a feature width cannot be correct at one size and timed at another, because
#:   there is one pool and both gates read it;
#: * widening the workflow's range is a change to the matrix, which the correctness suite picks up
#:   in the same commit -- not a second list here that drifts from it silently.
#:
#: At cp=16 the ladder is N_token 2048 / 4096 / 8192 -> M = 262144 / 1048576 / 4194304, crossed
#: with all four feature widths: twelve cells, the largest of which (4194304 x 512) is 4.3 GB in
#: and 4.3 GB out. N_token=12288 is excluded by the matrix itself; see its ``workflow_M`` facet.
_WORKFLOW_M = tuple(
    v for v in LN_DUAL_GATED.axis("M").values if LN_DUAL_GATED.axis("M").facets["workflow_M"](v)
)
_WORKFLOW_D = tuple(
    v for v in LN_DUAL_GATED.axis("N").values if LN_DUAL_GATED.axis("N").facets["workflow_D"](v)
)

#: The two cells that are NOT workflow shapes and are kept anyway: ``4096x256x128`` is the
#: launch-bound regime, where per-call host cost is the whole measurement, and ``301x200x256`` is
#: off-grid on both extents. A suite that timed only production shapes would stop noticing either.
_OFF_LADDER_CELLS = [(4096, 256, 128), (301, 200, 256)]

_FRONT_CELLS = [(m, d, d) for m in _WORKFLOW_M for d in _WORKFLOW_D] + _OFF_LADDER_CELLS


@pytest.mark.parametrize("M,N,K", _FRONT_CELLS, ids=lambda v: str(v))
@matrix_exempt(
    "the shape family is a LIST OF CELLS, not an axis product: each entry is one workflow shape "
    "with its own pinned median, and a cross product of M x N x K would be hundreds of timed runs "
    "pinning numbers nothing runs at. The matrix governs the kernel's configuration axes, which "
    "the other tests in this file do draw from it"
)
def test_the_fused_front_projection_shapes(M, N, K):
    """The shapes the fused TriMul front projection runs at, in the preferred weight layout.

    Deliberately the SAME cell list as the unfused gate's, so the two files' pins subtract.
    """
    ops = _operands(M, N, K)
    fn = _call(ops, tile_N=256, chunk_g=16)
    fn()
    torch.cuda.synchronize()
    # Two projections of width N, each M x N x K MACs. The LayerNorm's own O(M*K) work is not
    # counted: it is not a matrix multiply, and folding it in would inflate the TFLOP/s figure with
    # arithmetic no GEMM roofline covers.
    _bench(("front", M, N, K), fn, flops=2 * 2 * M * N * K)


@LN_DUAL_GATED.parametrize(
    "chunk_g",
    only={"chunk_g": (1, 16)},
    because=(
        "8 is the declared unsupported value, so there is nothing to time; the two supported "
        "layouts are the comparison this cell exists to make"
    ),
)
def test_the_weight_layout_costs_what_it_costs(chunk_g):
    """The two layouts compute bit-identical results, so any time difference IS the layout's cost.

    ``chunk_g == 1`` interleaves the two weights on the host on every call; ``chunk_g == 16`` loads
    them directly through a two-tensor TMA and does no host work at all. The interleave is inside
    the timed region deliberately -- it is a real per-call cost of choosing that layout, and hiding
    it would make the comparison meaningless.
    """
    M, N, K = 4096, 512, 256
    ops = _operands(M, N, K)
    fn = _call(ops, tile_N=256, chunk_g=chunk_g)
    fn()
    torch.cuda.synchronize()
    _bench(("layout", chunk_g), fn, flops=2 * 2 * M * N * K)


@LN_DUAL_GATED.parametrize("blk_k")
def test_each_k_tile_width_costs_what_it_costs(blk_k):
    """The K tile swept at one shape: fewer, wider k-tiles against more, narrower ones.

    Unlike every other sweep in this file the cells are NOT bit-identical to each other -- the tile
    reassociates the sums. So this measures a family of related kernels, and what it pins is that
    the default (64) stays the fast one. If a narrow tile ever won, the front door's ``auto_blk_k``
    policy would be leaving performance on the table at every K it applies to.
    """
    M, N, K = 4096, 512, 256
    ops = _operands(M, N, K)
    fn = _call(ops, tile_N=256, chunk_g=1, blk_k=blk_k)
    fn()
    torch.cuda.synchronize()
    _bench(("blk_k", blk_k), fn, flops=2 * 2 * M * N * K)


@LN_DUAL_GATED.parametrize(
    "ln_bias",
    "mask",
)
def test_each_epilogue_and_prologue_term_costs_what_it_costs(ln_bias, mask):
    """The optional terms swept at ONE shape, so the differences are the terms and nothing else.

    The LayerNorm bias is a PROLOGUE term and the mask an EPILOGUE one, swept together because they
    are the two whose absence is compiled out rather than made zero. Their costs should be
    independent; a cell where the pair costs more than the sum of the singles would mean one of them
    is changing the pipeline depth rather than just adding work.
    """
    M, N, K = 4096, 512, 256
    ops = _operands(M, N, K)
    nb = torch.randn(K, device="cuda", dtype=torch.float32) if ln_bias else None
    mk = torch.rand(M, device="cuda", dtype=torch.bfloat16) if mask else None
    fn = _call(ops, tile_N=256, chunk_g=16, norm_bias=nb, mask=mk)
    fn()
    torch.cuda.synchronize()
    _bench(("terms", ln_bias, mask), fn, flops=2 * 2 * M * N * K)


@LN_DUAL_GATED.parametrize("gate3")
def test_the_fused_output_gate_costs_what_it_costs(gate3):
    """The output gate against its own absence, at the shape the TriMul workflow uses it at.

    The design claim is that an output-gate work tile is an ORDINARY work tile of a wider operand.
    So the expected cost is the ratio of tile counts and nothing else: with ``2N = 512`` and
    ``N3 = 256`` the operand grows by half, and a cost much above that would mean the gate is
    perturbing the dual tiles rather than merely following them.

    Pinned on the element-interleave layout because that is the only layout that carries the gate --
    the block layout hands the kernel two separate weight tensors, so there is nothing to append to.
    """
    M, N, K = 4096, 256, 256
    ops = _operands(M, N, K, N3=K if gate3 else 0)
    fn = _call(ops, tile_N=256, chunk_g=1)
    fn()
    torch.cuda.synchronize()
    n3 = K if gate3 else 0
    _bench(("gate3", gate3), fn, flops=2 * M * K * (2 * N + n3))


#: The two shapes the algebraic fold's gap against the upstream was measured at, and the two that
#: bracket it: the excess is a per-output-tile cost, so it is worst where the mainloop is shortest
#: (K=128, two k-tiles) and least where it is longest (K=512, eight). Pinning only one of them would
#: pin a number whose size is an accident of K.
_FOLD_CELLS = [(262144, 128, 128), (262144, 512, 512)]


@pytest.mark.parametrize("M,N,K", _FOLD_CELLS, ids=lambda v: str(v))
@LN_DUAL_GATED.parametrize("fusion_variant")
def test_each_fusion_variant_costs_what_it_costs(fusion_variant, M, N, K):
    """Both fusions at the same shape, so the difference IS the fusion and not the geometry.

    The two compute the same thing by different routes -- ``prolog_ln`` normalizes A in shared
    memory and multiplies, ``alg_fold`` multiplies the RAW A and repairs the accumulator rank-one --
    so they are NOT expected to agree, and this does not assert that they do. What it holds is each
    one's own cost, which is what a bring-back needs: the fold was ported from an upstream kernel it
    is byte-identical to, and the whole point of the port is that it also costs what that kernel
    costs.

    **Why these cells specifically.** `alg_fold` measured +7.4% of device cycles against the
    upstream at K=128 and +2.3% at K=512 (2026-08-12, after `140442c`). Most of that was the price
    of compiling N as a symbolic extent where the upstream bakes it, so its out-of-bounds bounds
    folded to literals while ours predicated per element.

    **That is no longer the case: K and N are both baked now**, which cost nothing -- in this
    workflow ``n == k == D``, so the two are the same feature dimension and the artifact count is
    unchanged, and the measured cold compile did not move (9 of 12 cells got slightly faster).
    Re-measured front-door to front-door at matched tiles: **+5.1% at D=128 falling monotonically
    to +1.2% at D=512**, where `prolog_ln` and the two-A path both reach parity (0.99-1.01).

    So a residual remains, and its SHAPE says where it is not: the work-tile count is identical at
    both ends (16384), while the k-tiles per tile go 2 -> 8, so a gap that shrinks with D is a fixed
    per-work-tile cost -- epilogue or per-tile setup, not the mainloop. These pins hold that
    residual; lowering them is an improvement, not a fix.

    ``chunk_g=1`` on both: it is the layout the fold was measured and byte-compared at, and the only
    one the TriMul workflow's gate3 path can use.
    """
    ops = _operands(M, N, K)
    fn = _call(ops, tile_N=256, chunk_g=1, fusion_variant=fusion_variant)
    fn()
    torch.cuda.synchronize()
    _bench(("variant", fusion_variant, M, N, K), fn, flops=2 * 2 * M * N * K)


@pytest.mark.parametrize("M,N,K", _FOLD_CELLS, ids=lambda v: str(v))
@LN_DUAL_GATED.parametrize(
    "x_gate",
    drop={"x_gate": (False,)},
    because=(
        "False is the one-activation kernel, whose cost the fusion-variant gate above already "
        "pins at these exact cells -- timing it twice would pin one number in two places"
    ),
)
def test_the_two_activation_gate_costs_what_it_costs(x_gate, M, N, K):
    """The two-A kernel's own cost, at the cells the one-activation fold is pinned at.

    Same shapes as `test_each_fusion_variant_costs_what_it_costs` deliberately: the two entries then
    subtract, so what the second activation COSTS is readable rather than inferred. It is not free
    and should not look it -- a pipeline stage holds four operand buffers here instead of two, so
    the mainloop runs at half the depth, and there are two WGMMA streams with a full drain between
    a k-tile and its stage release.

    **This pin holds a cost at PARITY with the upstream, which is what the port owes.** It did not
    start there: kernel-to-kernel (both precompiled, so neither pays its host-side fold inside the
    timed window) this path measured 1.0205x-1.0456x while N was a symbolic extent, and
    0.9841x-1.0035x with N baked -- the whole difference. K and N are now both baked, and the
    front-door measurement agrees: 0.9887x-1.0052x across D in {128, 256, 384, 512}.

    Baking cost nothing to buy that. In this workflow ``n == k == D``, so the two extents are the
    same feature dimension and a caller sees one artifact per width either way; the measured cold
    compile did not move.

    ``tile_N`` tiles the OUTPUT width here, not the 2N pre-activation -- but it is NOT simply ``N``:
    the SM90 WGMMA atom accepts at most 256 columns per warpgroup, so ``N`` itself is illegal at the
    512-wide cell. The pick is the upstream's own -- the largest multiple of 32 at or below 128 that
    divides N -- which is what its auto-pick would choose and therefore what the parity measurement
    was taken at.
    """
    hi = min(N, 128)
    tile_N = next(t for t in range(hi - hi % 32, 0, -32) if N % t == 0)
    ops = _operands(M, N, K, x_gate=x_gate)
    fn = _call(ops, tile_N=tile_N, chunk_g=1, fusion_variant="alg_fold")
    fn()
    torch.cuda.synchronize()
    # Two N-wide GEMMs, which is the same product as the dual's one 2N-wide one -- so this number
    # is directly comparable to the fusion-variant gate's at the same cell.
    _bench(("xgate", M, N, K), fn, flops=2 * 2 * M * N * K)


@LN_DUAL_GATED.parametrize("pingpong")
def test_the_ping_pong_schedule_costs_what_it_costs(pingpong):
    """Ping-pong against cooperative on the fold, at a shape where the epilogue is a real fraction.

    Swept on ``alg_fold`` only -- it is the sole variant that accepts the schedule, since
    ``prolog_ln`` refuses it at the front door (its per-row reduction spans every MMA warpgroup).
    That is not a narrowing of the axis, which is why there is no ``because=``: BOTH declared values
    are timed, just on the one variant that can run them.

    Two MMA warpgroups alternating mainloop and epilogue should win wherever one warpgroup's
    epilogue can hide under the other's MMA, and the two produce BIT-IDENTICAL output -- so any
    difference here is schedule and nothing else. It is pinned because the schedule's payoff was
    itself a bug once: ping-pong measured 72% SLOWER than the upstream's until the mainloop's k-loop
    was given its static trip count, and a pin is what would have caught that without a two-tree
    comparison.
    """
    M, N, K = 262144, 128, 128
    ops = _operands(M, N, K)
    # tile_N=128, not the 256 the rest of this file uses: ping-pong at tile_M=128 caps the CTA tile
    # N at 208 (the two-warpgroup schedule's register budget), so 256 is REFUSED at the front door.
    # 128 is also the tile every ours-vs-upstream comparison of this variant was made at.
    fn = _call(ops, tile_N=128, chunk_g=1, fusion_variant="alg_fold", pingpong=pingpong)
    fn()
    torch.cuda.synchronize()
    _bench(("pingpong", pingpong), fn, flops=2 * 2 * M * N * K)


@matrix_exempt("gates the host submit path, which is shape-independent; there is nothing to sweep")
def test_the_fused_entry_dispatch_cost_holds(dev):
    """The PER-CALL PYTHON cost of the fused entry holds, with the CPU divided out.

    Purpose
        The timed cells above use ``mode="device"``, so they report the kernel and nothing about the
        Python that submits it. This entry has MORE per-call host work than the unfused sibling --
        the LayerNorm operands' dtype and rank checks, the K-tile choice, and (on the
        element-interleave layout) the weight interleave -- so it is the one whose submit path most
        needs its own gate.

    Semantics
        A TINY shape, because dispatch is shape-independent per entry point. Calibrated against the
        bare ``GemmSm90`` launch, so the pinned quantity is this entry's cost RELATIVE to a plain
        CuTe-DSL submit -- a dimensionless ratio that survives a change of host, where the
        microseconds do not.

    Args:
        dev: The CUDA device fixture.

    Returns:
        None.
    """
    ops = _operands(8, 8, 16)
    call = _call(ops, tile_M=64, tile_N=32, chunk_g=1)
    call()
    torch.cuda.synchronize()
    assert_host_dispatch(
        call, ("dispatch", "layernorm_dual_gated_gemm"), PINS, __file__, device=dev
    )
