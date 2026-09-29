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

"""PERF GATE: ``fold_cp_ops.kernels.dual_gated_gemm.dual_gated_gemm``.

**Do NOT lock the GPU clock for this gate**, and do not harvest its pins with one either.

Locking makes any single measurement far more repeatable, but it SHIFTS THE MEAN: an H100's
free-running clock follows the 700 W power cap and hence the workload, so a locked pin and a
free-running measurement are samples of two different distributions. Mixing them produced 23
failures in one suite, none of them real. Both sides run free-running, and the sample counts
(``calibration.N_SAMPLES_PIN_TIME`` / ``N_SAMPLES_TEST_TIME``) are what make the medians converge --
1/sqrt(N), measured.

Pins a median (ms/call) per cell and fails if the measured median drifts above the cell's band.

**Read this beside ``test_benchmark_perf_gemm.py``.** That gate times the same SM90 mainloop under
the default epilogue; this one times it under the gated fold, at the SAME contraction. The dual-gated
kernel does ONE pass over A to produce both projections, so the comparison that matters is against
*two* plain GEMMs of half the width -- which is the alternative it exists to replace. A regression
that shows here and not there is in the fold; one that shows in both is in the mainloop.

**The layout axis is swept at a fixed shape.** ``chunk_g`` changes only how the 2N weight is laid
out and which registers the fold pairs -- the arithmetic is bit-identical (asserted in
``tests/kernels/test_dual_gated_gemm.py``), so any time difference between the two is purely the
cost of the layout, which is what these cells isolate. The host-side weight interleave that
``chunk_g == 1`` needs is INSIDE the timed region on purpose: it is a per-call cost that a caller
choosing that layout actually pays.

Timing goes through ``bench_utils.benchmark_single(mode="device")`` per CLAUDE.md -- never a
hand-rolled ``perf_counter`` loop around an async launch, which measures host dispatch and can
report a bandwidth above the physical link.

Harvest new pins (prints medians and TFLOP/s, skips the assertion)::

    CPO_PERF_MEASURE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s \
        tests/perf/test_benchmark_perf_dual_gated_gemm.py
"""

import os

import pytest
import torch

from benchmark.distributed import bench_utils as BU
from tests.perf.calibration import assert_cell, assert_host_dispatch
from tests.perf.pins import load as load_pins
from fold_cp_ops.kernels.dual_gated_gemm import dual_gated_gemm
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from tests.kernels.test_dual_gated_gemm import DUAL_GATED  # the ONE declaration, imported

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(
    _SM != 9, reason=f"pins are SM90 (H100) numbers; this GPU is sm_{_SM}0"
)

_MEASURE = os.environ.get("CPO_PERF_MEASURE", "0") == "1"

#: This gate is reference-calibrated, so it may run in a shared session: each cell divides its
#: measurement by a reference measured in the SAME session, and each carries its own measured
#: spread in the pin file rather than picking between two hand-drawn tolerance constants.

_ROUNDS, _WARMUP, _ITERS = 30, 10, 10


#: Pinned medians (ms/call) on an idle H100 80GB HBM3, bf16, k-major operands, persistent
#: non-pingpong grid, tile_M 128. An ABSENT key measures and skips rather than failing, which is
#: what makes adding a cell cheap: write it, run the harvest command, paste the number.
#:
#: **The layout pair is the set to read first.** ``chunk_g=1`` costs 0.0262 ms against
#: ``chunk_g=16``'s 0.0190 -- 1.38x, at a shape where the two compute bit-identical results. All of
#: that difference is the per-call host weight interleave the element-interleave layout needs, which
#: is why the front door recommends the block layout and why the interleave is inside the timed
#: region. It is also an independent confirmation of the upstream heuristic that picked
#: ``chunk_g=16`` uniformly.
#:
#: **The epilogue block reads as a set**, all at M/N/K = 4096/512/256 so the differences are the
#: terms and nothing else: no terms 0.0192 ms, a post-gate mask +10%, both projection biases +19%,
#: and the two together +29%. The mask is a column-vector load and a multiply; the bias pair is two
#: half-width row-vector loads and two adds, so costing about twice the mask is the expected shape.
#:
#: **The tile sweep is flat** (0.0190-0.0194 ms across 32/64/128/256), which is the useful negative
#: result: at this shape the kernel is not tile-sensitive, so a future regression that shows up at
#: one tile only is a real geometry bug rather than a tuning artifact.
#:
#: The three front-projection cells run at 334-472 TFLOP/s. The K=128 cell is the slow one for the
#: same structural reason as the plain GEMM's: two 64-wide k-tiles leave the mainloop almost no
#: steady state to amortize its prologue over. That is what the front projection IS -- watch whether
#: the number moves, not whether it approaches the back einsum's.
#: **Re-harvested, and the reason is two effects that were being read as one.**
#:
#: 1. `dual_gated_gemm` is now `@autotune(gate="do_autotune")`-decorated, so a fixed-config call
#:    passes through one wrapper frame. Measured across these cells that costs **+0.3 to +1.1 us**,
#:    about 1-3%. Small, real, and bought for a real thing: ONE entry point per kernel instead of
#:    three. Recorded rather than argued with, exactly as the front-door checks were.
#: 2. **The old pins were already at or over their band**, independently of that. Harvested at the
#:    pre-decoration commit on an idle box, ``('epi', 'both', False)`` measured 0.0221-0.0236 ms
#:    against a 0.0214 pin -- a 6.8% spread whose top is +10.3%, i.e. already outside the 10% band.
#:    It passed only because the cells before it in the file warm the GPU; run as a ``-k`` subset
#:    from a cold 345 MHz idle clock it failed 2 runs in 8 at BOTH commits.
#:
#: So the numbers below are the **max of three consecutive full-file runs**, not a single median.
#: A pin set at a lucky low is a gate that fails on the unlucky run, which teaches people to re-run
#: it rather than to read it -- the same reasoning that now gives each cell its own measured band.
#: This gate's pinned numbers, loaded from the sibling JSON of the SAME NAME. The name is the
#: documentation: a number in a failure message is traceable to its file without grepping, and a
#: pin file names the one test that consumes it. Enforced in conftest -- a timed perf module that
#: does not expose `PINS` FAILS rather than skips, because a gate inventing its own number format
#: is exactly what that check exists to catch.
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
        AssertionError: If the drift-corrected median exceeds the pin by more than its own band.
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


def _operands(M, N, K, dtype=torch.bfloat16):
    """Build one cell's operands, pitch-padded so any N is legal.

    Args:
        M: Token extent.
        N: Output feature extent -- the pre-activation is twice this.
        K: Contraction extent (the model's hidden width).
        dtype: Operand and output element type.

    Returns:
        ``(A, Wg, Wp, out)`` on CUDA. The output's row pitch is padded up to the 16-byte floor, so
        an off-grid N is timed rather than refused.
    """
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    Wg = torch.randn(N, K, device="cuda", dtype=dtype) * 0.1
    Wp = torch.randn(N, K, device="cuda", dtype=dtype) * 0.1
    e = 16 // dtype.itemsize
    out = torch.empty(M, (N + e - 1) // e * e, device="cuda", dtype=dtype)[:, :N]
    return A, Wg, Wp, out


#: The A2A-fused TriMul workflow's OWN shapes, not a convenient ladder.
#:
#: In that workflow the front projection consumes a local token shard of ``N_token x N_token/cp``
#: rows, so ``M = N_token**2 / cp``, and both projections are ``(D, D)`` so ``K == N == D``. The
#: N_token values are the ones ``profiling/distributed/trimul/dlc_composite_k_dtensor_cp_le16``
#: actually runs -- 2048 through 12288 at cp 2..16 -- and D spans all four TriMul feature widths.
#:
#: **Why these and not the whole product.** cp=2 at N_token=12288 is M = 75.5M rows, which at D=512
#: is 77 GB for the activation alone: not a cell, an OOM. The ladder below takes cp=16 (the smallest
#: shard, hence the largest N_token that fits) at each end of the range and crosses it with every D,
#: then adds the single largest feasible cell. The two small cells are kept: `4096x256x128` is the
#: launch-bound regime and `301x200x256` is off-grid, and a suite that only measured production
#: shapes would stop noticing per-call cost entirely.
#:
#: M                  N_token  cp    what it is
#: 262144             2048     16    smallest production shard
#: 1048576            4096     16
#: 4194304            8192     16
_WORKFLOW_M = tuple(
    v for v in DUAL_GATED.axis("M").values if DUAL_GATED.axis("M").facets["workflow_M"](v)
)
_WORKFLOW_D = tuple(
    v for v in DUAL_GATED.axis("N").values if DUAL_GATED.axis("N").facets["workflow_D"](v)
)

#: Not workflow shapes, and kept anyway: 4096x256x128 is the launch-bound regime where per-call
#: host cost IS the measurement, and 301x200x256 is off-grid on both extents.
_OFF_LADDER_CELLS = [(4096, 256, 128), (301, 200, 256)]

#: DERIVED from the matrix, not listed -- and it must stay the same construction as
#: `test_benchmark_perf_layernorm_dual_gated_gemm.py`'s, because the difference between the two
#: files' pins at a shared cell IS the LayerNorm fusion's cost. A cell present in one and absent
#: from the other is not a missing measurement, it is a subtraction that silently stops working.
_FRONT_CELLS = [(m, d, d) for m in _WORKFLOW_M for d in _WORKFLOW_D] + _OFF_LADDER_CELLS


@pytest.mark.parametrize("M,N,K", _FRONT_CELLS, ids=lambda v: str(v))
@matrix_exempt(
    "the shape family is a LIST OF CELLS, not an axis product: each entry is one workflow shape "
    "with its own pinned median, and a cross product of M x N x K would be hundreds of timed runs "
    "pinning numbers nothing runs at. The matrix governs the kernel's configuration axes, which "
    "the other tests in this file do draw from it"
)
def test_dual_gated_gemm_front_projection_shapes(M, N, K):
    """The shapes the fused TriMul front projection runs at, in the preferred weight layout."""
    A, Wg, Wp, out = _operands(M, N, K)
    tile_n = 128
    fn = lambda: dual_gated_gemm(  # noqa: E731
        A, Wg, Wp, out, tile_M=128, tile_N=tile_n, chunk_g=16
    )
    fn()
    torch.cuda.synchronize()
    # Two projections of width N, each M x N x K MACs.
    _bench(("front", M, N, K), fn, flops=2 * 2 * M * N * K)


@DUAL_GATED.parametrize(
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
    them directly. The interleave is inside the timed region deliberately -- it is a real per-call
    cost of choosing that layout, and hiding it would make the comparison meaningless.
    """
    M, N, K = 4096, 512, 256
    A, Wg, Wp, out = _operands(M, N, K)
    fn = lambda: dual_gated_gemm(  # noqa: E731
        A, Wg, Wp, out, tile_M=128, tile_N=128, chunk_g=chunk_g
    )
    fn()
    torch.cuda.synchronize()
    _bench(("layout", chunk_g), fn, flops=2 * 2 * M * N * K)


@DUAL_GATED.parametrize(
    "bias",
    "mask",
    drop={"bias": ("gate_only",)},
    because="the half-biased cell is refused at the front door, so there is nothing to time",
)
def test_each_epilogue_term_costs_what_it_costs(bias, mask):
    """The epilogue terms swept at ONE shape, so the differences are the terms and nothing else."""
    M, N, K = 4096, 512, 256
    A, Wg, Wp, out = _operands(M, N, K)
    bg = bp = None
    if bias == "both":
        bg = torch.randn(N, device="cuda", dtype=torch.float32)
        bp = torch.randn(N, device="cuda", dtype=torch.float32)
    mk = torch.rand(M, device="cuda", dtype=torch.bfloat16) if mask else None
    fn = lambda: dual_gated_gemm(  # noqa: E731
        A, Wg, Wp, out, tile_M=128, tile_N=128, chunk_g=16, bg=bg, bp=bp, mask=mk
    )
    fn()
    torch.cuda.synchronize()
    _bench(("epi", bias, mask), fn, flops=2 * 2 * M * N * K)


@DUAL_GATED.parametrize(
    "tile_n",
    because="sweeps the whole declared tile pool at one shape to show the geometry's cost curve",
    only={"tile_n": (32, 64, 128, 256)},
)
def test_the_tile_geometry_costs_what_it_costs(tile_n):
    """The tile pool swept at one shape: a frozen tile would hide a regression at the others."""
    M, N, K = 4096, 512, 256
    A, Wg, Wp, out = _operands(M, N, K)
    fn = lambda: dual_gated_gemm(  # noqa: E731
        A, Wg, Wp, out, tile_M=128, tile_N=tile_n, chunk_g=16
    )
    fn()
    torch.cuda.synchronize()
    _bench(("tile", tile_n), fn, flops=2 * 2 * M * N * K)


# ── the autotuned entry: does sweeping the weight layout pay for itself? ───────────────────────
# The gated kernel tunes ONE knob, and the two values are not close: at the shape below,
# `chunk_g=1` costs 0.0253 ms and `chunk_g=16` costs 0.0180 ms -- a 1.4x spread, because the
# element-interleave layout runs a host-side interleave kernel on EVERY call. So unlike the plain
# GEMM's pool, this one has no ties, and the pick is stable enough to assert directly.
#
# The comparison is against the better of the two FIXED layout cells pinned above, which is exactly
# "did the tuner find the layout a developer would have picked by reading the pins".

#: Cells for the tuned entry: ``(M, N, K)`` -> the fixed :data:`PINS` key it must match or beat.
_AUTOTUNE_CELLS = {
    (4096, 512, 256): ("layout", 16),
    (262144, 256, 128): ("front", 262144, 256, 128),
}

#: The layout the tuner is expected to choose. Both cells are 16-aligned in K and N, so both
#: candidates are admissible and the choice is decided by measurement rather than by the
#: alignment floor -- which is what makes this assertion about the TUNER and not about `validity`.
_AUTOTUNE_EXPECTED_CHUNK_G = 16


@matrix_exempt(
    "sweeps the AUTOTUNER over a shape list; the kernel matrix declares correctness cells, and "
    "the tuned entry compiles both layouts per cell"
)
@pytest.mark.parametrize("M,N,K", list(_AUTOTUNE_CELLS))
def test_dual_gated_autotune_picks_the_block_interleave_and_beats_the_fixed_default(M, N, K):
    """The tuner chooses `chunk_g=16` and lands within the pinned band of the fixed cell.

    **The pick is asserted directly here, unlike the plain GEMM's**, and the difference is measured
    rather than stylistic: the GEMM's top configs tie within the timer's noise, so pinning one is a
    coin flip. These two layouts are 1.4x apart, because `chunk_g=1` runs a host-side weight
    interleave on every call. A tuner that stopped choosing 16 at an aligned shape would be broken,
    not unlucky.

    The timing is a DIRECT `dual_gated_gemm` call with the winning `chunk_g`, for the same reason as
    the GEMM gate: the tuner's wrapper costs ~12 us of Python per call and ``_config=`` about 2 us,
    which at a launch-bound shape is a double-digit fraction of the kernel. Timing through either
    would report a dispatch cost as a bad pick. What this cell is responsible for is the CONFIG the
    sweep chose; the wrapper's own cost is gated in the GEMM file, where it is one number rather than
    one per kernel.

    Args:
        M: Token count.
        N: Per-projection output width.
        K: Contraction (feature) extent.
    """
    A, Wg, Wp, out = _operands(M, N, K)
    dual_gated_gemm(A, Wg, Wp, out, tile_M=128, tile_N=128, do_autotune=True)
    torch.cuda.synchronize()

    won = dual_gated_gemm.autotuner.best_config
    assert won["chunk_g"] == _AUTOTUNE_EXPECTED_CHUNK_G, (
        f"autotune chose chunk_g={won['chunk_g']} at ({M}, {N}, {K}); both layouts are admissible "
        f"here and the block interleave is ~1.4x faster, so this is a tuner regression rather than "
        f"a tie broken the other way"
    )

    fn = lambda: dual_gated_gemm(  # noqa: E731
        A, Wg, Wp, out, tile_M=128, tile_N=128, chunk_g=won["chunk_g"]
    )
    fn()
    torch.cuda.synchronize()
    median_ms = BU.benchmark_single(
        fn, mode="device", rounds=_ROUNDS, warmup=_WARMUP, iters=_ITERS
    ).median_ms

    # Gated on the FIXED cell's pin, so `emit=False`: harvesting here would overwrite that cell's
    # pin with the TUNED number, after which the tuner is compared against its own previous output.
    assert_cell(
        lambda: median_ms,
        _AUTOTUNE_CELLS[(M, N, K)],
        PINS,
        __file__,
        emit=False,
        describe=lambda _: (
            f"autotune ({M}, {N}, {K}) chunk_g={won['chunk_g']} vs the FIXED cell. A sweep that "
            f"lands slower than the hardcoded default is worse than not sweeping."
        ),
    )


@matrix_exempt("a property of this module's own call sites; nothing is launched")
def test_the_fixed_cells_do_not_route_through_the_tuner():
    """Every pinned fixed-config cell calls `dual_gated_gemm` directly, so its number is a kernel's.

    A pinned cell that reached the tuner would time a sweep on its first call and the winner
    thereafter -- two different numbers from one assertion, depending on collection order.
    """
    import inspect

    for fn in (
        test_dual_gated_gemm_front_projection_shapes,
        test_the_weight_layout_costs_what_it_costs,
        test_each_epilogue_term_costs_what_it_costs,
        test_the_tile_geometry_costs_what_it_costs,
    ):
        src = inspect.getsource(fn)
        assert "do_autotune" not in src, (
            f"{fn.__name__} sets do_autotune; a pinned cell must measure a kernel, never a sweep. "
            f"The gate defaults to off, so omitting it is all that is required."
        )


@matrix_exempt("gates the host submit path, which is shape-independent; there is nothing to sweep")
def test_the_dual_gated_entry_dispatch_cost_holds(dev):
    """The PER-CALL PYTHON cost of the `dual_gated_gemm` entry holds, with the CPU divided out.

    Purpose
        Since the timed cells moved to ``mode="device"`` they report the kernel alone, so the submit
        path needs its own gate. This entry is the one worth watching most closely of the three: it
        carries the `chunk_g` weight-layout decision and the gated-fold epilogue arguments, so it has
        the most per-call Python to accumulate.

    Semantics
        A TINY shape, because dispatch is shape-independent per entry point. Calibrated against the
        bare ``GemmSm90`` launch, so the pinned quantity is this entry's cost RELATIVE to a plain
        CuTe-DSL submit -- a ratio that survives a change of host, where the microseconds do not.
    """
    A, Wg, Wp, out = _operands(8, 8, 8)
    call = lambda: dual_gated_gemm(A, Wg, Wp, out, tile_M=64, tile_N=64)  # noqa: E731
    call()
    torch.cuda.synchronize()
    assert_host_dispatch(call, ("dispatch", "dual_gated_gemm"), PINS, __file__, device=dev)
