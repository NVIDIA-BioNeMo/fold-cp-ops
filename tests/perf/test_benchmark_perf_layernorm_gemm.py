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

"""PERF GATE: ``fold_cp_ops.kernels.layernorm_gemm.layernorm_gemm`` on a single SM90 GPU.

**Do NOT lock the GPU clock for this gate**, and do not harvest its pins with one either. Locking
makes any single measurement far more repeatable but SHIFTS THE MEAN -- an H100's free-running clock
follows the 700 W power cap and hence the workload -- so a locked pin compared against a
free-running run is a comparison of two distributions. Both sides run free-running and the sample
counts in ``calibration`` are what make the medians converge.

Pins one median (ms/call) per ``(M, D, dtype)`` cell and fails if the measured median drifts above
that cell's own measured band. A regression gate, not a benchmark report.

**The grid is keyed on what the kernel sees.** This is a single-device kernel: it takes an ``(M, D)``
activation and a ``(D, D)`` weight and knows nothing about the distributed mesh, so keying on
``cp`` would couple the gate to a grid it cannot observe. The production row counts that mesh
implies are covered directly, as ``M``.

**The default path is timed, not a pinned tile.** ``select="heuristic"`` is what a caller gets, so
it is what a regression would reach: a change that moves the heuristic's pick shows up here, where a
hard-pinned tile would hide it. The heuristic runs on the host and adds one dict lookup.

**The optional epilogue terms are part of the grid, not folded away.** ``gate3`` adds a whole
``(M, D)`` C operand -- a TMA descriptor, a staging buffer and an epilogue multiply -- so a cell
without it does not bound a cell with it. One gated cell per k-tiling regime is enough to hold that
line without doubling the harvest.

Timing goes through the shared CUDA-event harness (``bench_utils.benchmark_single(mode="device")``)
per CLAUDE.md -- never a hand-rolled ``perf_counter`` loop, which measures host dispatch rather than
device time. ``mode="device"`` is correct here because the kernel is per-rank-independent and
contains no collective; ``mode="event"`` would read the host's launch cadence at the launch-bound
cells.

Harvest new pins (prints medians, skips the assertion)::

    CPO_PERF_MEASURE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s tests/perf/

then merge the printed ``PINHARVEST`` lines into the sibling JSON with ``pins.merge_harvest``.
Harvest from ONE WHOLE-SUITE run, not per file: sustained load moves the free-running clock, so a
pin taken on an otherwise idle GPU is measured under conditions the suite never reproduces.
"""

from __future__ import annotations

import math

import pytest
import torch

from benchmark.distributed import bench_utils as BU
from fold_cp_ops.kernels.layernorm_gemm import layernorm_gemm
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from tests.kernels.test_layernorm_gemm import LAYERNORM_GEMM  # the ONE declaration, imported
from tests.perf.calibration import assert_cell, assert_host_dispatch
from tests.perf.pins import load as load_pins

# ── SM90 gate ──────────────────────────────────────────────────────────────────────────────────
_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(
    _SM != 9, reason=f"pins are SM90 (H100) numbers; this GPU is sm_{_SM}0"
)

#: Sampling counts for one cell's median. This kernel is compute-bound at the production widths and
#: launch-pressured at the small ones, so the two ends have very different spreads; each cell's own
#: pinned ``rel_std`` sets its band rather than one constant covering both.
_ROUNDS, _WARMUP, _ITERS = 50, 10, 10

# ── the grid ───────────────────────────────────────────────────────────────────────────────────
# Four things this list is built to hold, none of which a power-of-two ladder would:
#
# 1. **The production shapes.** The A2A-fused TriMul back-half calls this kernel at feature widths
#    128/256/384/512 with M = N_token**2 / cp. A gate that stopped at M=4096 would never see the
#    multi-wave persistent regime where the per-CTA statistics scratch is reused across work tiles.
# 2. **Both k-tiling regimes.** 128/256/384/512 all take the 64-wide k-tile; 136/200/264 are 8 mod
#    16, so the mainloop runs the 16-wide tile with a predicated partial LAST tile. That path has
#    four times the k-loop trip count at a comparable width, and nothing about the aligned cells
#    bounds it -- it is the whole reason this kernel exists as the workflow's fallback.
# 3. **Both schedules.** The size heuristic returns ping-pong only at D == 128 above its M floor,
#    so the two cells at (1048576, 128) and (262144, 128) are what keep that branch measured; every
#    other production cell exercises the cooperative bulk tile.
# 4. **The launch-bound end.** At M = 512 the kernel is submit-dominated, which is where a change to
#    the host path (an extra dict lookup, a re-derived fold) shows up and where a compute-bound cell
#    is blind.
_CELLS = [
    # --- production: the TriMul back-half's widths at production token counts ---
    (1048576, 128, torch.bfloat16),  # heuristic picks ping-pong here
    (262144, 128, torch.bfloat16),  # heuristic picks ping-pong here
    (1048576, 256, torch.bfloat16),
    (262144, 256, torch.bfloat16),
    (262144, 384, torch.bfloat16),
    (262144, 512, torch.bfloat16),
    # --- the 16-wide k-tile: 8 mod 16, partial last k-tile, 4x the trip count ---
    (262144, 136, torch.bfloat16),
    (262144, 200, torch.bfloat16),
    (262144, 264, torch.bfloat16),
    # --- the exact 32- and 16-wide tiles: 16-aligned but not 64 ---
    (262144, 192, torch.bfloat16),
    (262144, 320, torch.bfloat16),
    # --- float16, which shares the atom but not the rounding, at both k-tilings ---
    (262144, 256, torch.float16),
    (262144, 200, torch.float16),
    # --- the launch-bound end ---
    (4096, 128, torch.bfloat16),
    (512, 128, torch.bfloat16),
]

#: The cells timed WITH the fused output gate. A separate list because the gate is a functional
#: variant -- it adds a C operand and therefore a different compiled kernel -- rather than a
#: performance knob, and because pinning every cell twice would double a harvest that already runs
#: at production token counts. One per k-tiling regime holds the line.
_GATED_CELLS = [
    (262144, 256, torch.bfloat16),
    (262144, 200, torch.bfloat16),
]

#: This gate's pinned numbers, loaded from the sibling JSON of the SAME NAME. Enforced in the perf
#: conftest -- a timed perf module that does not expose ``PINS`` FAILS rather than skips, because a
#: gate inventing its own number format is exactly what that check exists to catch.
PINS = load_pins(__file__)


def _cell_id(M: int, D: int, dt: torch.dtype, gated: bool = False) -> str:
    """Stable pytest id / pin-key suffix for one cell.

    Args:
        M: Token count. Any positive int.
        D: Feature width. Any positive int meeting the 16-byte floor.
        dt: Element dtype; only its short name is used, so ``torch.`` is stripped.
        gated: Whether the fused output gate is present. It is part of the id because it selects a
            different compiled kernel, so two cells differing only in it must not share a pin.

    Returns:
        A string like ``M262144xD384-bfloat16``, safe for a pytest node id (no spaces or dots).
    """
    return f"M{M}xD{D}-{str(dt).replace('torch.', '')}" + ("-gate3" if gated else "")


def _flops(M: int, D: int) -> int:
    """Multiply-accumulate FLOPs in one call: the ``(M, D) @ (D, D)`` contraction.

    Purpose
        Turns an opaque millisecond figure into a TFLOP/s comparable to the SM90 roofline, which is
        what makes a pinned number readable to someone who did not harvest it.

    Semantics
        Counts the GEMM alone. The LayerNorm reduction, the rank-one repair and the gate are all
        O(M*D) against the GEMM's O(M*D^2), so at the production widths they are a rounding error on
        this figure -- and folding them in would make the derived rate depend on which optional
        terms a cell carries, which is precisely what a comparable number must not do.

    Args:
        M: Token count.
        D: Feature width, which is both the contraction extent and the output width.

    Returns:
        ``2 * M * D * D`` -- one multiply and one add per contracted element.
    """
    return 2 * M * D * D


def _make_fn(M: int, D: int, dt: torch.dtype, device: torch.device, gated: bool = False):
    """Build a warmed zero-argument launch closure for one cell.

    Allocation and the host-side weight fold are folded into the closure's captured state where they
    can be, and the JIT compile is warmed before returning, so the timed region is the front door
    plus the kernel rather than a first-call compile.

    **The front door's own host work IS timed, deliberately.** ``layernorm_gemm`` re-folds the weight
    on every call, which is a real per-call cost a caller pays unless they hoist it -- pulling it out
    of the timed region here would pin a number no caller can observe.

    Args:
        M: Token count; must be positive.
        D: Feature width; must meet the 16-byte floor for `dt`.
        dt: Activation and weight dtype -- float16 or bfloat16.
        device: CUDA device to allocate on. Must be the device the timing runs on, or the
            measurement times a cross-device launch.
        gated: Whether to build and pass the ``(M, D)`` per-element output gate.

    Returns:
        A zero-argument callable running one `layernorm_gemm`, already warmed and synchronized.
    """
    x = torch.randn(M, D, device=device, dtype=dt)
    w = torch.randn(D, device=device, dtype=torch.float32)
    b = torch.randn(D, device=device, dtype=torch.float32)
    B = torch.randn(D, D, device=device, dtype=dt) / math.sqrt(D)
    g3 = torch.randn(M, D, device=device, dtype=dt) if gated else None
    out = torch.empty(M, D, device=device, dtype=dt)

    def run():
        layernorm_gemm(x, w, B, bias=b, eps=1e-6, out=out, gate3=g3)

    for _ in range(10):  # warm the JIT compile and settle clocks
        run()
    torch.cuda.synchronize(device)
    return run


@LAYERNORM_GEMM.parametrize(
    "M",
    "D",
    "input_dtype",
    cells=_CELLS,
    because=(
        "a perf gate pins one median per cell, so this grid is a LIST of cells rather than a cross "
        "product -- the full matrix would be hundreds of timed runs at production token counts. "
        "Every component is still checked against the declared pool, so the shapes that are timed "
        "and the shapes that are correctness-tested cannot drift apart."
    ),
)
def test_layernorm_gemm_perf(dev, M, D, input_dtype):
    """The measured median stays within this cell's own band of its pinned SM90 median.

    Args:
        dev: Session device fixture, pinned to this rank's GPU (see the perf conftest).
        M: Token count for this cell.
        D: Feature width for this cell.
        input_dtype: Activation dtype for this cell.

    Raises:
        AssertionError: If the measured median exceeds the pin by more than its band.
        pytest.skip.Exception: If the cell is unpinned; the median is printed so it can be
            harvested.
    """
    fn = _make_fn(M, D, input_dtype, dev)
    measure = lambda: (  # noqa: E731
        BU.benchmark_single(
            fn,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            iters=_ITERS,
            device=dev,
            mode="device",
            label=f"layernorm_gemm M={M} D={D} {input_dtype}",
        ).median_ms
    )
    flops = _flops(M, D)
    assert_cell(
        measure,
        (M, D, str(input_dtype).replace("torch.", "")),
        PINS,
        __file__,
        describe=lambda ms: f"{_cell_id(M, D, input_dtype)} {flops / (ms * 1e-3) / 1e12:.1f} TF/s",
    )


@LAYERNORM_GEMM.parametrize(
    "M",
    "D",
    "input_dtype",
    cells=_GATED_CELLS,
    because=(
        "the SUBJECT is the fused output gate's marginal cost, and the gate is a FUNCTIONAL variant "
        "-- a whole (M, D) C operand, its TMA descriptor and its staging buffer -- so it compiles to "
        "a different kernel that the un-gated cells do not bound. One cell per k-tiling regime holds "
        "that line; pinning every cell twice would double a harvest that already runs at production "
        "token counts."
    ),
)
def test_layernorm_gemm_gated_perf(dev, M, D, input_dtype):
    """The fused output gate's cell holds its own pin, separately from the un-gated one.

    The gate is fused into the epilogue precisely so it costs no extra pass over DRAM. That claim is
    only true while the C load overlaps the mainloop, and a change that serialized it would leave
    every un-gated cell untouched -- which is why the gated cells are pinned rather than assumed.

    Args:
        dev: Session device fixture.
        M: Token count for this cell.
        D: Feature width for this cell.
        input_dtype: Activation dtype for this cell.

    Raises:
        AssertionError: If the measured median exceeds the pin by more than its band.
        pytest.skip.Exception: If the cell is unpinned.
    """
    fn = _make_fn(M, D, input_dtype, dev, gated=True)
    measure = lambda: (  # noqa: E731
        BU.benchmark_single(
            fn,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            iters=_ITERS,
            device=dev,
            mode="device",
            label=f"layernorm_gemm gate3 M={M} D={D} {input_dtype}",
        ).median_ms
    )
    flops = _flops(M, D)
    assert_cell(
        measure,
        (M, D, str(input_dtype).replace("torch.", ""), "gate3"),
        PINS,
        __file__,
        describe=lambda ms: (
            f"{_cell_id(M, D, input_dtype, True)} {flops / (ms * 1e-3) / 1e12:.1f} TF/s"
        ),
    )


@matrix_exempt("gates the host submit path, which is shape-independent; there is nothing to sweep")
def test_the_layernorm_gemm_entry_dispatch_cost_holds(dev):
    """The PER-CALL PYTHON cost of the `layernorm_gemm` entry holds, with the device time divided out.

    Purpose
        Every cell above is timed with ``mode="device"``, which reports the kernel and says nothing
        about the Python that submits it. This entry does MORE host work than most: it validates
        eight arguments, resolves a configuration and RE-FOLDS the weight on every call, so an
        unwatched regression here is one a caller pays at every token count and no cell would see.

    Semantics
        A TINY shape, because dispatch cost is shape-independent per entry point -- at 64 rows the
        kernel is a rounding error on the submit path, which is exactly the ratio being pinned.
        Calibrated against a bare ``GemmSm90`` launch: what the reference must share with the target
        is the DISPATCH class (CuTe-DSL / TVM-FFI submit), not the arithmetic, and the device side is
        divided out by construction.

    Args:
        dev: Session device fixture, pinned to this rank's GPU.

    Raises:
        AssertionError: If the ratio to the bare-launch reference exceeds its pinned band.
        pytest.skip.Exception: If the pin is absent; the ratio is printed so it can be harvested.
    """
    call = _make_fn(64, 128, torch.bfloat16, dev)
    torch.cuda.synchronize()
    assert_host_dispatch(call, ("dispatch", "layernorm_gemm"), PINS, __file__, device=dev)
