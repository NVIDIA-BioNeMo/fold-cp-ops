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

"""PERF GATE: ``fold_cp_ops.workflows.trimul_autotune.trimul_autotuned`` -- the WHOLE cp=1 TriMul workflow.

**Why this file exists, stated plainly: its absence hid a 1.425x regression while every per-kernel
gate stayed green.** This package had a pinned perf gate for `gemm`, `gemm_hadamard`, `layernorm`,
`dual_gated_gemm`, `layernorm_dual_gated_gemm` and the rest -- and NONE for the dispatcher that
composes them. The regression lived in the composition, not in any kernel: op 7 was routed to the
low-level `gemm` entry with a hand-pinned ``(128, 128)`` tile from a local helper that `main` has no
counterpart for, while `main` routes it through its TUNED dispatch. Every kernel was individually at
parity and the workflow was 1.425x at N_token=2048 and 1.453x at 4096.

A per-kernel matrix cannot see this. The kernels were not slower; the dispatcher chose differently.
So the subject here is the CHAIN -- one number per workflow cell, end to end.

**Do NOT lock the GPU clock for this gate**, and do not harvest its pins with one. Locking makes a
single measurement repeatable but SHIFTS THE MEAN: an H100's free-running clock follows the 700 W
cap and hence the workload, so a locked pin and a free-running measurement sample two different
distributions. Both sides run free-running; `calibration`'s sample counts make the medians converge.

**Timing goes through ``bench_utils.benchmark_single(mode="device")``** per CLAUDE.md, never a
hand-rolled ``perf_counter`` loop -- a raw launch is async, so ``perf_counter`` would time host
dispatch. ``mode="device"`` (adaptive per-rank rep count) is correct here because the cp=1 workflow
is collective-free; the distributed A2A-fused path must use ``mode="event", reduce="max"`` instead,
and that is a different gate.

**The cells that do not fit at cp=1, and why that is not a coverage hole to fix here.** 18 of the
32 declared cells raise ``OutOfMemoryError`` on an 80 GB H100 -- N_token 8192 and 12288 at every D,
plus 4096 at D=512. That is a property of the SHAPE, not of this harness: at cp=1 the dispatcher is
handed the UN-SHARDED token extent, so one cell needs

    x                 B*N*N*D*2
    staging buf       2*D*(B*N*N)*2      <- twice x, and the binding term
    tri (D*B, N, N)   D*B*N*N*2

which is 16 + 32 + 16 = 64 GiB before any output, against 79.2 GiB usable. **Measured single-tree,
not inferred:** 8192/128 and 4096/512 OOM with only ONE tree imported, so pairing two trees is not
what exhausts the card and splitting the harness would not recover them. They are reachable only in
the distributed path, where each rank holds ``N_token**2 / cp``, and they belong to that gate.

Every cell -- including the ones that do fit -- is guarded by a RUNTIME OOM skip and never by a
memory estimate. Two reasons, both load-bearing: an estimate cannot know the allocator's
fragmentation or the kernels' workspace, and a static skip would freeze TODAY's card into the
source, so the same cells would keep skipping on a 141 GB H200 that can actually run them.

**STATUS: UNHARVESTED.** Every cell is unpinned, so it PRINTS AND SKIPS rather than gating. That is
deliberate: this repo's pins come from ONE serial whole-perf-suite run, because sustained load moves
the free-running clock, and a gate pinned from a single-file run on a shared box would fail in the
real session for reasons unrelated to this workflow. Harvest with::

    CPO_PERF_MEASURE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s tests/perf/

and merge the ``PINHARVEST`` lines into the sibling JSON.
"""

import pytest
import torch

from benchmark.distributed import bench_utils as BU
from tests.perf.calibration import assert_cell, assert_host_dispatch
from tests.perf.pins import load as load_pins
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.workflows.trimul_autotune import trimul_autotuned
from tests.workflows.test_trimul_autotune import TRIMUL_AUTOTUNE  # the ONE declaration, imported

# ── SM90 gate ──────────────────────────────────────────────────────────────────────────────────
_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(
    _SM != 9, reason=f"pins are SM90 (H100) numbers; this GPU is sm_{_SM}0"
)

# Fewer rounds than a kernel gate: one cell here is a whole workflow (6 ms at the smallest cell,
# 130 ms at the largest), so the per-round cost is 10-100x a single-kernel cell and the same round
# count would put this file into the tens of minutes on its own.
_ROUNDS, _WARMUP, _ITERS = 15, 5, 5

PINS = load_pins(__file__)


def _inputs(N, D, direction, dev, B=1):
    """Build the workflow's operands and return a warmed zero-argument call.

    Args:
        N: The token extent (``N_token``). Must satisfy ``N % 8 == 0`` -- op 7's ``(N, N)`` operands
            need a 16-byte row pitch and the front door refuses otherwise, BY NAME. Every cell here
            meets it; a cell that did not would surface as a refusal, not a wrong number.
        D: The feature width, one of the four TriMul widths.
        direction: ``"outgoing"`` or ``"incoming"``. ``"sideways"`` is in the matrix as the invalid
            value the front door must reject and is not a perf cell.
        dev: The device fixture's device.
        B: The batch. Left at 1 because B only scales the flattened token axis ``M = B*N*N``, so a
            batched cell measures the same arithmetic at a larger M and buys the gate no regime it
            does not already have.

    Returns:
        A zero-argument closure, already invoked once so the compile AND the op-7 autotune sweep sit
        outside the timed region. Timing a cold call would measure the tuner, not the workflow.

    Raises:
        torch.cuda.OutOfMemoryError: Propagated to the caller, which converts it to a skip.
    """
    dt = torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(0)
    r = lambda *s, d=dt: torch.randn(*s, generator=g, device=dev, dtype=d)  # noqa: E731
    kw = dict(
        x=r(B, N, N, D),
        norm_in_w=r(D, d=torch.float32),
        norm_in_b=r(D, d=torch.float32),
        p_in_w=r(2 * D, D),
        g_in_w=r(2 * D, D),
        norm_out_w=r(D, d=torch.float32),
        norm_out_b=r(D, d=torch.float32),
        p_out_w=r(D, D),
        g_out_w=r(D, D),
        direction=direction,
    )
    call = lambda: trimul_autotuned(**kw)  # noqa: E731
    call()
    torch.cuda.synchronize()
    return call


# ── the workflow cells ─────────────────────────────────────────────────────────────────────────
# The full product of the two declared workflow facets against both valid directions. Unlike a
# kernel gate, this grid is NOT trimmed to a hand-picked list: the whole lesson of the regression
# this file exists for is that one cell cannot stand in for the ladder. At N_token=2048 the pinned
# (128, 128) op-7 tile and the tuned choice agree, so a 2048-only gate reads 0.99x and sees nothing;
# the same code is 1.45x at 4096. Two token counts x four widths x two directions is 16 cells, and
# the ones too large for the box skip at runtime.
_WORKFLOW_CELLS = [
    (n, d, direction)
    for n in (2048, 4096)
    for d in (128, 256, 384, 512)
    for direction in ("outgoing", "incoming")
]


@TRIMUL_AUTOTUNE.parametrize(
    "N",
    "D",
    "direction",
    cells=_WORKFLOW_CELLS,
    because=(
        "the perf cells are the A2A-fused workflow's OWN configuration -- the `workflow_N_token` "
        "and `workflow_D` facets crossed with both valid directions -- rather than the shape pools, "
        "which carry the alignment and off-grid regimes that correctness needs and perf does not. "
        "N_token 8192 and 12288 are in the ladder but are omitted here because at cp=1 the "
        "dispatcher holds the un-sharded token extent and all 16 such cells were MEASURED to raise "
        "OutOfMemoryError on an 80 GB card; they are reachable only in the distributed gate. "
        "'sideways' is the direction the front door must reject and is covered as an unsupported "
        "region in tests/workflows/test_trimul_autotune.py, not timed here."
    ),
)
def test_trimul_workflow_perf(N, D, direction, dev):
    """The whole cp=1 TriMul chain holds its pinned median at each workflow cell.

    This is deliberately ONE number for the entire workflow rather than a per-op breakdown. A
    breakdown would re-measure what the per-kernel gates already pin; what no other gate can see is
    the chain's total, which is the quantity that regressed when the dispatcher picked op 7's
    config differently from `main` while every kernel stayed at parity.
    """
    torch.manual_seed(0)
    # The OOM guard spans the MEASUREMENT, not just the build. A cell can fit for one call and still
    # exhaust the card under repeated timing: the harvest takes N_SAMPLES_PIN_TIME medians, each a
    # full `benchmark_single`, and the allocator fragments across them. Measured here --
    # (4096, 384, "outgoing") completed while (4096, 384, "incoming") raised inside the timed region
    # with 2.47 GiB free, and a guard around `_inputs` alone turned that into a FAILURE rather than
    # the skip it is. Sizing the guard to the build was the bug; the shape is not more supported
    # because its first call happened to fit.
    try:
        call = _inputs(N, D, direction, dev)
        measure = lambda: (  # noqa: E731
            BU.benchmark_single(
                call, mode="device", rounds=_ROUNDS, warmup=_WARMUP, iters=_ITERS
            ).median_ms
        )
        assert_cell(measure, ("workflow", N, D, direction), PINS, __file__)
    except torch.cuda.OutOfMemoryError:
        # Freed before the skip so the NEXT cell does not inherit this one's fragmentation and
        # cascade into a skip it would otherwise not need.
        call = measure = None
        torch.cuda.empty_cache()
        pytest.skip(
            f"N_token={N} D={D} {direction} does not fit at cp=1 on this card: the front's staging "
            f"buffer alone is {2 * D * N * N * 2 / 2**30:.1f} GiB, before op 7's (N, N) operands. "
            f"A runtime skip, never a memory-estimate gate -- an estimate cannot know the "
            f"allocator's fragmentation or the kernels' workspace, which is what decides it here."
        )


@matrix_exempt(
    "the subject is the per-call HOST cost of the dispatcher itself, which is the same Python for "
    "every shape -- parametrizing it over the shape pools would pin the same number many times"
)
def test_trimul_dispatch_host_cost(dev):
    """The dispatcher's per-call host cost holds, as a ratio to a bare kernel launch.

    Why this cell is not optional here. Every timed cell above uses ``mode="device"``, which reports
    the kernel and says nothing about the Python that submits it -- and this entry has MORE of that
    Python than any single kernel does: it runs the size->config heuristic, builds the config,
    reshapes and transposes operands into views, and makes four separate kernel calls. On a small
    shape that host cost is the whole runtime, which is precisely where a dispatcher regression
    hides from a device-time gate.

    The shape is the smallest the front door accepts (``N % 8 == 0``, a declared ``D``), so the
    device work is negligible and what remains is dispatch. The pin is a dimensionless RATIO to a
    bare launch measured back-to-back in the same session, which is what makes it survive the load
    and the machine that an absolute millisecond figure does not.
    """
    torch.manual_seed(0)
    call = _inputs(8, 32, "outgoing", dev)
    assert_host_dispatch(call, ("dispatch", "trimul_autotuned"), PINS, __file__, device=dev)
