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

"""PERF GATE: the bare ``GemmSm90`` mainloop on a single SM90 GPU.

**Do NOT lock the GPU clock for this gate**, and do not harvest its pins with one either.

Locking makes any single measurement far more repeatable, but it SHIFTS THE MEAN: an H100's
free-running clock follows the 700 W power cap and hence the workload, so a locked pin and a
free-running measurement are samples of two different distributions. Mixing them produced 23
failures in one suite, none of them real. Both sides run free-running, and the sample counts
(``calibration.N_SAMPLES_PIN_TIME`` / ``N_SAMPLES_TEST_TIME``) are what make the medians converge --
1/sqrt(N), measured.

Pins a median (ms/call) per ``(M, N, K, L, tile_M, tile_N)`` cell and fails if the measured median
drifts above a band DERIVED FROM ITS OWN MEASURED SPREAD. A regression gate, not a
benchmark report.

**Why time the base functor with no epilogue.** This is the floor every GEMM-derived kernel in the
tree is built on -- ``GemmDefaultSm90``, ``DualGatedGemmStagedSm90``, and the A2A-fused stores are
all the same mainloop under a different store. Timing it alone separates "the mainloop got slower"
from "the epilogue got more expensive", which a gate on the full entry cannot do. The paired gate on
the host entry lives in ``tests/perf/test_benchmark_perf_gemm.py``.

Timing goes through the shared CUDA-event harness (``bench_utils.benchmark_single(mode="device")``)
per CLAUDE.md -- never a hand-rolled ``perf_counter`` loop, which would measure host dispatch rather
than device time. Clocks are locked for the session by the autouse ``clock_locked`` fixture, which
warns and proceeds where clock control is denied.

Harvest new pins (prints medians and the achieved TFLOP/s, skips the assertion)::

    CPO_PERF_MEASURE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s \
        tests/perf/test_benchmark_perf_gemm_sm90.py

then merge the printed ``PINHARVEST`` lines into the sibling JSON::

    python tests/perf/pins.py <captured log> "an idle H100 80GB HBM3, clocks unlocked"

A pin is ``(median_ms, relative spread)``; see ``tests/perf/calibration.py`` for why the
spread is pinned rather than a band being chosen.
"""

import os

import pytest
import torch

from benchmark.distributed import bench_utils as BU
from tests.perf.pins import load as load_pins
from tests.perf.calibration import assert_cell
from tests.kernels.test_gemm_sm90 import (  # the ONE declaration, imported
    GEMM_SM90,
    _build_operands,
    run_base_gemm,
)

# ── SM90 gate ──────────────────────────────────────────────────────────────────────────────────
_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(
    _SM != 9, reason=f"pins are SM90 (H100) numbers; this GPU is sm_{_SM}0"
)

_MEASURE = os.environ.get("CPO_PERF_MEASURE", "0") == "1"

#: This gate's entry point IS the host reference, so it has no dispatch gate of its own.
#:
#: `calibration.make_host_reference` times a bare `GemmSm90` launch at a tiny shape -- exactly
#: what `run_base_gemm` does here. Gating that against itself is the degenerate case: reference
#: and target would be independent samples of ONE quantity, so `corrected = now / (now / pin)`
#: is `pin` for every input on every machine, INCLUDING when the submit path genuinely
#: regresses, because the regression moves both sides and the drift factor absorbs it. The gate
#: would pass forever while looking entirely reasonable.
#:
#: So this entry's dispatch cost is not gated here -- it is the DENOMINATOR the other three
#: gates' dispatch pins are expressed in, which is what makes those pins portable.
DISPATCH_EXEMPT = (
    "this gate's entry IS the host reference (calibration.make_host_reference times a bare "
    "GemmSm90 launch), so a dispatch gate here would compare the reference against itself and "
    "could never fail"
)


_ROUNDS, _WARMUP, _ITERS = 30, 10, 10


#: The back einsum at production N_token, at the tile that is BEST FOR A SINGLE DEVICE.
#:
#: **Deliberately NOT the distributed kernel's config.** `main`'s `_resolve_back_config` returns
#: (128, 128) and its harvested H100+IB table pins (128, 128) c(1,1) for every (D, cp0, cp1) key --
#: but that choice carries constraints this gate does not have. cluster_M must be 1 there because
#: the pe_aligned per-peer M-tiling and M-multicast are mutually exclusive, and the tile interacts
#: with the partial-N A2A store. None of that applies to a single-device GEMM, so pinning the
#: single-device gate to it would measure the distributed path's compromises and call them the
#: kernel's speed.
#:
#: What the difference costs, measured per cell (cluster (1, 1), which the runner fixes):
#:
#:     N_token  L   best tile      TFLOP/s   vs (128,128)
#:      2048    8   (256, 208)       716        1.39x
#:      2048   16   (256, 192)       639        1.25x
#:      2048   32   (256, 208)       636        1.50x
#:      4096    8   (256, 128)       655        1.30x
#:      4096   16   (256, 192)       688        1.52x
#:      8192    8   (256, 192)       721        2.14x
#:     12288    8   (256, 208)       708        2.88x
#:
#: **Read the TFLOP/s column, not the ratio column.** At its own best tile the kernel holds
#: 636-721 TF across the entire production range -- FLAT. An earlier version of this block pinned
#: (128, 128) and reported a rate that "halved between 4096 and 12288"; that halving was the tile,
#: not the kernel, and stating it as a property of the kernel would have been wrong. tile_N is the
#: whole effect: at N=8192 the L2 rasterization swizzle moves it under 15% and cluster (2, 1)
#: recovers 349 -> 557 TF, while widening the tile alone reaches 744.
#:
#: L = Dloc * B, so the family is {8, 16, 32, 48}: Dloc = D/cp in {8, 16} for D in {128, 256} at
#: cp in {8, 16}, times B in {1, 2}.
_TRIMUL_BACK_PRODUCTION = [
    # (M, N, K, L, tile_M, tile_N) at the per-shape best single-device tile
    (2048, 2048, 2048, 8, 256, 208),  # Dloc=8,  B=1
    (2048, 2048, 2048, 16, 256, 192),  # Dloc=8,  B=2  /  Dloc=16, B=1
    (2048, 2048, 2048, 32, 256, 208),  # Dloc=16, B=2
    (4096, 4096, 4096, 8, 256, 128),
    (4096, 4096, 4096, 16, 256, 192),
    (8192, 8192, 8192, 8, 256, 192),
    (12288, 12288, 12288, 8, 256, 208),
]

#: The original four back cells, at tile (128, 256). Kept because the (128, 256) / (128, 128)
#: pair at 2048 is the one direct read of what the wider N tile buys this einsum.
_TRIMUL_BACK_WIDE_TILE = [
    (1000, 1000, 1000, 16, 128, 256),  # N_token=1000, D=128 at cp=8
    (1000, 1000, 1000, 48, 128, 256),  # N_token=1000, D=384 at cp=8
    (2048, 2048, 2048, 16, 128, 256),  # N_token=2048, D=128 at cp=8
    (2048, 2048, 2048, 32, 128, 256),  # N_token=2048, D=256 at cp=8
]

#: Every cell that IS the back einsum. An explicit list rather than a slice of `_CELLS`, so a
#: future square cell that is NOT the einsum can be added without weakening the squareness guard.
_TRIMUL_BACK = _TRIMUL_BACK_WIDE_TILE + _TRIMUL_BACK_PRODUCTION


def _measure_geometry(M, N, K, L):
    """Rounds / warmup / iters for one cell, scaled by how much work it is.

    Purpose
        The production back-einsum cells are 25-120 ms per launch. At the default 30x10 that is
        7-36 SECONDS of wall clock per cell, which would make this gate something people skip --
        and a gate that is skipped protects nothing.

    Semantics
        A cell costing more than `_HEAVY_FLOP` drops to 5 rounds / 2 warmup / 1 iter. Fewer samples
        is acceptable HERE specifically because these cells are deeply compute-bound: the launch
        overhead they would be averaging out is ~14 us against a 120 ms kernel, i.e. 0.01%. It would
        NOT be acceptable for the launch-bound cells, which is why the rule is keyed on the work and
        not applied uniformly.

    Args:
        M: Rows of A / D.
        N: Columns of B / D.
        K: Contraction extent.
        L: Batch extent.

    Returns:
        ``(rounds, warmup, iters)``.
    """
    if 2 * L * M * N * K >= _HEAVY_FLOP:
        return 5, 2, 1
    return _ROUNDS, _WARMUP, _ITERS


#: Work above which a cell is measured with fewer, longer samples. 4e12 FLOP is ~6 ms at this
#: kernel's rate, comfortably above every launch-bound cell and below the 4096-at-L=16 cell.
_HEAVY_FLOP = 4e12


# ── the grid: what the mainloop's speed actually depends on ────────────────────────────────────
# Three regimes, because a GEMM's bottleneck moves and a single-size gate would only ever watch one:
#
# 1. **Compute-bound** -- large M/N/K, where the WGMMA pipeline is saturated and the number that
#    matters is TFLOP/s. A regression here is a broken pipeline or a lost stage.
# 2. **Memory-bound** -- thin K, where the kernel is reading operands faster than it multiplies
#    them. A regression here is usually a lost TMA multicast or a shrunken stage count.
# 3. **Launch/tail-bound** -- small M, where the wave quantization and the epilogue drain dominate.
#    This is the regime the TriMul front's small-token cells sit in.
#
# Tile shapes are varied WITH the shape rather than swept independently: the point of the gate is
# that each shape keeps the speed it had at the tile the dispatcher would pick for it.
_CELLS = [
    # (M, N, K, L, tile_M, tile_N) -- compute-bound
    (4096, 2048, 2048, 1, 128, 256),
    (4096, 2048, 2048, 1, 128, 128),
    (4096, 2048, 2048, 1, 256, 128),
    (1001, 2048, 2048, 1, 128, 256),  # odd M at the same work: measures the M-tail cost
    # memory-bound: thin K
    (4096, 2048, 64, 1, 128, 256),
    (4096, 2048, 128, 1, 128, 128),
    # launch/tail-bound: small M, batched
    (128, 256, 256, 7, 128, 128),
    (301, 200, 192, 2, 128, 128),  # the off-grid correctness shape, timed
    (1, 8, 8, 1, 64, 64),  # the degenerate floor: pure launch overhead
    # --- the A2A-fused TriMul BACK einsum, which is what this base kernel is subclassed for ---
    # out[b,i,j,d] = sum_k a[b,i,k,d] . b[b,j,k,d] with i = j = k = N_token, so M == N == K and
    # L = B * (D/cp). `test_trimul_back_cells_are_square` below enforces that; a decoupled K would
    # measure an O(N^2) thin-K matmul and silently flip every compute-vs-comm conclusion drawn
    # from it (CLAUDE.md's K = N_token HARD RULE). Both lists are defined above.
    *_TRIMUL_BACK_WIDE_TILE,
    *_TRIMUL_BACK_PRODUCTION,
]


@GEMM_SM90.parametrize(
    "M",
    "N",
    "K",
    "L",
    "tile_M",
    "tile_N",
    cells=_CELLS,
    because=(
        "a perf gate pins ONE number per cell, so the grid is a list and not a product -- the "
        "cross product of these six axes is over 100k timed launches. The nine cells span the "
        "three regimes a GEMM moves between (compute-bound, thin-K memory-bound, small-M "
        "launch-bound), which is what a single-size gate cannot see; correctness coverage of the "
        "full pools stays in tests/kernels/test_gemm_sm90.py."
    ),
)
def test_gemm_sm90_perf(M, N, K, L, tile_M, tile_N):
    """The bare mainloop holds its pinned median at each cell.

    Compiles once outside the timed region (``run_base_gemm`` caches per configuration and the
    warmup rounds hit that cache), so what is timed is the launch and the device work, never the
    JIT. ``mode="device"`` with a fixed iteration count is the CUDA-event path from ``bench_utils``;
    a ``perf_counter`` loop around an async launch would measure host dispatch and report a
    fictitious speed.
    """
    torch.manual_seed(0)
    A, B, D = _build_operands(M, N, K, L, torch.bfloat16, "k", "k")
    run_base_gemm(A, B, D, tile_M, tile_N, (1, 1), False, True)  # compile + correctness warmup
    torch.cuda.synchronize()

    rounds, warmup, iters = _measure_geometry(M, N, K, L)
    measure = lambda: (
        BU.benchmark_single(  # noqa: E731
            lambda: run_base_gemm(A, B, D, tile_M, tile_N, (1, 1), False, True),
            mode="device",
            rounds=rounds,
            warmup=warmup,
            iters=iters,
        ).median_ms
    )
    flop = 2 * L * M * N * K
    assert_cell(
        measure,
        (M, N, K, L, tile_M, tile_N),
        PINS,
        __file__,
        flops=flop,
        describe=lambda ms: f"({flop / (ms * 1e-3) / 1e12:6.1f} TFLOP/s)",
    )


#: Pinned medians (ms/call) harvested on an idle H100 80GB HBM3, bf16, k-major operands, persistent
#: non-pingpong grid, cluster (1,1). {(M, N, K, L, tile_M, tile_N): ms}. An absent key measures and
#: skips rather than failing, so a new cell does not turn red before it has ever been harvested.
#:
#: The three regimes are visible in the numbers and are worth reading as a set. The compute-bound
#: cells reach 712 TFLOP/s at tile (128, 256) -- about 72% of the H100's dense bf16 peak -- and 529
#: at (128, 128), which is what the wider N tile buys. The thin-K cells sit at 93-186 TFLOP/s: the
#: pipeline is starved, not slow. The small cells all land within a few percent of 0.011 ms, which
#: is the per-call floor of the event-window harness rather than anything about the kernel -- they
#: gate against a launch-path regression, not against arithmetic.
#:
#: The TriMul back-einsum cells are the set to read as a curve rather than as points. At tile
#: (128, 256) they sit at 366-642 TFLOP/s: the N_token=2048 pair is genuinely compute-bound, the
#: N_token=1000 pair is not yet -- 1000 is under three 384-wide N tiles, so the grid does not fill.
#:
#: At the per-shape BEST SINGLE-DEVICE tile, across the production N_token the profiled grid runs:
#:
#:     N_token=2048   0.211 ms   650 TFLOP/s   (L=8, tile (256, 208))
#:     N_token=4096   1.702 ms   646 TFLOP/s   (L=8, tile (256, 128))
#:     N_token=8192  13.190 ms   667 TFLOP/s   (L=8, tile (256, 192))
#:     N_token=12288 43.474 ms   683 TFLOP/s   (L=8, tile (256, 208))
#:
#: **FLAT** -- 646-683 TF, about 65-69% of the H100's dense bf16 peak, all the way to 12288. That is
#: the useful statement for the fused workflow, and it is the OPPOSITE of what this block said when
#: it pinned the distributed kernel's (128, 128): there the same sweep read 521 -> 245 TF and looked
#: like the kernel degrading at scale. It was the tile.
#:
#: The L family is Dloc * B, and it is flat in L at fixed N_token (650/640/637 TF at L=8/16/32,
#: N_token=2048) -- the batch is pure grid, so a future L-dependence is a scheduler bug.
#: This gate's pinned numbers, loaded from the sibling JSON of the SAME NAME. The name is the
#: documentation: a number in a failure message is traceable to its file without grepping, and a
#: pin file names the one test that consumes it. Enforced in conftest -- a timed perf module that
#: does not expose `PINS` FAILS rather than skips, because a gate inventing its own number format
#: is exactly what that check exists to catch.
PINS = load_pins(__file__)


@GEMM_SM90.parametrize(
    "M",
    "N",
    "K",
    cells=[c[:3] for c in _TRIMUL_BACK],
    because=(
        "this is a property of the back-einsum cells specifically, so it sweeps exactly those "
        "three axes over exactly those cells; the other perf cells are not the einsum and are not "
        "required to be square."
    ),
)
def test_trimul_back_cells_are_square(M, N, K):
    """The back-TriMul cells contract over ``K == N_token``, never a decoupled K.

    CLAUDE.md makes this a HARD RULE, and the reason is that violating it does not fail -- it
    measures a *different kernel shape*. The back half is the square einsum ``out[b,i,j,d] =
    sum_k a[b,i,k,d] . b[b,j,k,d]`` with ``i = j = k = N_token``, i.e. O(N^3) and compute-bound at
    scale. Pin it with, say, K=256 instead and you are timing an O(N^2) thin-K matmul that is
    comm-bound, which silently inverts every compute-vs-comm, overlap-efficiency and crossover
    conclusion anyone draws from the numbers.

    Cheap to state, so it is stated as a test rather than a comment: a comment cannot fail when
    someone shrinks K to make the gate run faster.
    """
    assert M == N == K, (
        f"back-TriMul cell ({M}, {N}, {K}) is not square: the einsum's contraction dim IS N_token"
    )
