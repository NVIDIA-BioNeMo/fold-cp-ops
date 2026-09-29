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

"""PERF GATE: ``fold_cp_ops.kernels.gemm_layernorm_gemm.gemm_layernorm_gemm`` on one SM90 GPU.

**Do NOT lock the GPU clock for this gate**, and do not harvest its pins with one either. Locking
makes any single measurement far more repeatable but it SHIFTS THE MEAN -- an H100's free-running
clock follows the 700 W power cap and hence the workload -- so a locked pin compared against a
free-running measurement is a comparison between two different distributions. Both sides run
free-running; sampling is what makes the medians converge.

**The grid is what the entry point actually sees: ``(N, D, schedule)``.** ``N`` is the token extent
and ``D`` the feature extent, which together fix all three kernels' geometry. The third component
is the ``cont_pipe`` schedule spelled as a word, because a bool round-trips through JSON as ``true``
and a pin key that reads ``[128, 128, true]`` is one nobody can grep for.

**Why cells above the selection corner are here.** The dispatcher picks this chain only in the
launch-tiny corner ``N^2 * D <= 8.4e6``, so most of the grid is outside the regime it is chosen for.
They are pinned anyway: the corner is a heuristic, heuristics move, and a shape that is one edit
away from being selected must not be the shape nobody ever timed.

**Both schedules are pinned at two shapes.** ``cont_pipe`` is the one perf lever this chain has, and
its size heuristic is a claim about which side wins where. A gate that only ever timed the heuristic's
own pick could not tell a regression in one schedule from the heuristic quietly picking the other.

Timing goes through the shared CUDA-event harness (``bench_utils.benchmark_single``) per CLAUDE.md,
never a hand-rolled ``perf_counter`` loop, which would measure host dispatch rather than device time.

Harvest new pins (prints medians, skips the assertion)::

    CPO_PERF_MEASURE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s \
        tests/perf/test_benchmark_perf_gemm_layernorm_gemm.py

then merge the printed ``PINHARVEST`` lines into the sibling JSON with ``pins.merge_harvest``.
"""

from __future__ import annotations

import pytest
import torch

from benchmark.distributed import bench_utils as BU
from fold_cp_ops.kernels.gemm_layernorm_gemm import gemm_layernorm_gemm
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from tests.kernels.test_gemm_layernorm_gemm import GEMM_LAYERNORM_GEMM  # the ONE declaration
from tests.perf.calibration import assert_cell, assert_host_dispatch
from tests.perf.pins import load as load_pins

# ── SM90 gate ──────────────────────────────────────────────────────────────────────────────────
_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(
    _SM != 9, reason=f"pins are SM90 (H100) numbers; this GPU is sm_{_SM}0"
)

#: Rounds / warmup / iters for ``benchmark_single``. This chain launches THREE kernels per call and
#: the first is O(N^3), so a cell is far more expensive than a single-kernel gate's; the counts are
#: correspondingly lower and the per-cell band is what carries the strictness.
_ROUNDS, _WARMUP, _ITERS = 20, 5, 5

# ── the grid ───────────────────────────────────────────────────────────────────────────────────
# Read as (N, D, cont_pipe). Three things it is built to catch:
#
# 1. **The selected corner.** N^2*D <= 8.4e6 is where the dispatcher chooses this chain, and its
#    boundary cells (128/512 and 256/128, both exactly 8.4e6) are where a regression would first
#    change which variant wins the e2e comparison.
# 2. **Both predication paths of the projection.** D=264 is 8 mod 16, so no %16 divisor exists and
#    the output tile drops to 16 with a ceil-div tile count; D=208 is %16 but not %64. Neither is
#    reachable from a power-of-two ladder, and both are compiled in only on those shapes.
# 3. **Both token-pair schedules at one shape each side of the heuristic's crossover at 640.**
_CELLS = [
    # --- the selected launch-tiny corner (the heuristic picks the continuous schedule here) ---
    (128, 128, "cont"),
    (128, 256, "cont"),
    (128, 512, "cont"),  # exactly at the corner: 128^2 * 512 = 8.4e6
    (256, 128, "cont"),  # exactly at the corner: 256^2 * 128 = 8.4e6
    # --- above the corner, where a heuristic change would land ---
    (256, 256, "cont"),
    (256, 256, "perfeat"),  # the same shape on the other schedule
    (512, 512, "cont"),
    (512, 512, "perfeat"),
    (1024, 256, "perfeat"),  # past the crossover, so this IS the heuristic's pick
    # --- off-grid: N % 8 not % 64, and the two projection predication paths ---
    (200, 208, "cont"),
    (128, 264, "cont"),
]

#: This gate's pinned numbers, loaded from the sibling JSON of the SAME NAME. Enforced in conftest:
#: a timed perf module that does not expose ``PINS`` FAILS rather than skips.
PINS = load_pins(__file__)


def _cell_id(N: int, D: int, schedule: str) -> str:
    """Stable pytest id / pin-key suffix for a cell.

    Args:
        N: Token extent; any value from the matrix pool.
        D: Feature extent; likewise.
        schedule: ``"cont"`` or ``"perfeat"``.

    Returns:
        A string like ``N128xD512-cont``, safe for a pytest node id.
    """
    return f"N{N}xD{D}-{schedule}"


def _flops(N: int, D: int) -> float:
    """Multiply-adds x2 for one call: the token-pair einsum plus the output projection.

    Purpose:
        Turns an opaque millisecond figure into a TFLOP/s comparable to the SM90 roofline, which is
        what makes a pin readable years after it was harvested.

    Semantics:
        The einsum is ``O(N^3 * D)`` and the projection ``O(N^2 * D^2)``, so the first dominates
        wherever ``N > D``. The statistics reduction is ``O(N^2 * D)`` of adds and is omitted: it is
        two orders below either term across this grid and counting it would overstate the achieved
        rate rather than the reverse.

    Args:
        N: Token extent, positive.
        D: Feature extent, positive.

    Returns:
        Floating-point operations for one call at ``B == 1``, as a float.
    """
    return 2.0 * D * N**3 + 2.0 * (N * N) * D * D


def _make_fn(N: int, D: int, schedule: str, device: torch.device):
    """Build a warmed zero-argument launch closure for one cell.

    Allocation and the JIT compile are folded into the closure's captured state and warmed before
    returning, so the timed region is the three launches alone. A compile inside the timed region
    would dominate it by three orders of magnitude.

    Args:
        N: Token extent. Must be a multiple of 8.
        D: Feature extent. Must be a multiple of 8.
        schedule: ``"cont"`` or ``"perfeat"``; anything else is a typo in ``_CELLS`` and raises.
        device: The CUDA device to allocate on. Must be the device the timing runs on, or the
            measurement times a cross-device launch.

    Returns:
        A zero-argument callable running one full chain, already warmed and synchronized.

    Raises:
        ValueError: If ``schedule`` is not one of the two names.
    """
    if schedule not in ("cont", "perfeat"):
        raise ValueError(f"unknown schedule {schedule!r}; expected 'cont' or 'perfeat'")
    cont_pipe = schedule == "cont"
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(1, D, N, N, device=device, dtype=torch.bfloat16, generator=g) * 0.1
    b = torch.randn(1, D, N, N, device=device, dtype=torch.bfloat16, generator=g) * 0.1
    nw = torch.randn(D, device=device, dtype=torch.float32, generator=g)
    nb = torch.randn(D, device=device, dtype=torch.float32, generator=g)
    pw = (torch.randn(D, D, device=device, dtype=torch.float32, generator=g) * (D**-0.5)).to(
        torch.bfloat16
    )
    pb = torch.randn(D, device=device, dtype=torch.float32, generator=g)
    gate3 = torch.rand(N * N, D, device=device, dtype=torch.bfloat16, generator=g)

    def run():
        gemm_layernorm_gemm(a, b, nw, nb, pw, pb, gate3=gate3, eps=1e-5, cont_pipe=cont_pipe)

    for _ in range(3):  # warm the JIT compile and settle clocks
        run()
    torch.cuda.synchronize(device)
    return run


@GEMM_LAYERNORM_GEMM.parametrize(
    "N",
    "D",
    "cont_pipe",
    cells=[(n, d, s == "cont") for n, d, s in _CELLS],
    because=(
        "a perf gate pins one median per cell, so this grid is a LIST of cells rather than a cross "
        "product -- the full matrix would be hundreds of timed runs of an O(N^3) einsum. Every "
        "component is still checked against the declared pool, so the shapes that are timed and "
        "the shapes that are correctness-tested cannot drift apart."
    ),
)
def test_gemm_layernorm_gemm_perf(dev, N, D, cont_pipe):
    """The measured median stays within this cell's own band of its pinned H100 median.

    Args:
        dev: Session device fixture, pinned to this rank's GPU (see conftest).
        N: Token extent for this cell.
        D: Feature extent for this cell.
        cont_pipe: Which token-pair schedule this cell times.

    Raises:
        AssertionError: If the measured median exceeds the pin by more than its band.
        pytest.skip.Exception: If the cell is unpinned; the median is printed so it can be
            harvested. That is the intended state for a freshly landed gate.
    """
    schedule = "cont" if cont_pipe else "perfeat"
    fn = _make_fn(N, D, schedule, dev)
    measure = lambda: (  # noqa: E731
        BU.benchmark_single(
            fn,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            iters=_ITERS,
            device=dev,
            mode="device",
            label=f"gemm_layernorm_gemm N={N} D={D} {schedule}",
        ).median_ms
    )
    flops = _flops(N, D)
    assert_cell(
        measure,
        (N, D, schedule),
        PINS,
        __file__,
        flops=flops,
        describe=lambda ms: f"{_cell_id(N, D, schedule)} {flops / (ms * 1e-3) / 1e12:.1f} TFLOP/s",
    )


@matrix_exempt("gates the host submit path, which is shape-independent; nothing to sweep")
def test_the_entry_dispatch_cost_holds(dev):
    """The PER-CALL PYTHON cost of the ``gemm_layernorm_gemm`` entry holds, CPU divided out.

    Purpose
        The timed cells use ``mode="device"`` and so report the kernels alone. This entry does more
        host work per call than a single-kernel one -- it resolves the schedule heuristic, validates
        every argument, builds three compile-cache keys and launches three times -- and none of that
        is visible in a cell median. A regression there is invisible until it shows up as wall time
        in the e2e workflow.

    Semantics
        A TINY shape, because dispatch cost is shape-independent per entry point, calibrated against
        a bare launch: what the reference must share with the target is the DISPATCH class, not the
        arithmetic, since the device side is divided out by construction.

    Args:
        dev: Session device fixture.
    """
    call = _make_fn(72, 128, "perfeat", dev)
    torch.cuda.synchronize()
    assert_host_dispatch(call, ("dispatch", "gemm_layernorm_gemm"), PINS, __file__, device=dev)
