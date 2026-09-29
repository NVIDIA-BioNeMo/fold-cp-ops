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

"""PERF GATE: ``fold_cp_ops.kernels.gemm_hadamard.gemm_hadamard`` -- the fused per-element gate.

**Do NOT lock the GPU clock for this gate**, and do not harvest its pins with one. Locking makes a
single measurement repeatable but SHIFTS THE MEAN: an H100's free-running clock follows the 700 W
cap and hence the workload, so a locked pin and a free-running measurement sample two different
distributions. Both sides run free-running; the sample counts in `calibration` are what make the
medians converge.

**What this gate exists to watch, beyond "the GEMM got slower".** The Hadamard kernel differs from
the stock GEMM by one instruction in the epilogue and by an entire extra ``(l, m, n)`` TMA operand,
and only the second costs anything. Read against
``tests/perf/test_benchmark_perf_gemm.py``'s ``('epi', 'none', 'scalar')`` cell -- the stock GEMM
with a ``beta * C`` addend, i.e. the same C traffic under a different operator -- these cells say
whether the gate costs what reading C costs, or more. A regression that appears in both gates is in
the mainloop; one that appears only here is in the fused epilogue.

Timing goes through ``bench_utils.benchmark_single(mode="device")`` per CLAUDE.md -- never a
hand-rolled ``perf_counter`` loop around an async launch, which measures host dispatch.

**STATUS: UNHARVESTED except for the dispatch ratio.** Every timed cell below is unpinned, so it
PRINTS AND SKIPS rather than gating. That is deliberate and not an oversight: this repo's pins are
harvested from ONE serial whole-perf-suite run, because sustained load moves the free-running clock
(measured: the smallest launch-bound cell moves 11.8% between a quiet box and a loaded one), and a
gate pinned from a single-file run on a box shared with four other worktrees would fail in the real
session for reasons that have nothing to do with this kernel. Harvest with::

    CPO_PERF_MEASURE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s tests/perf/

and merge the ``PINHARVEST`` lines into the sibling JSON. The dispatch cell IS pinned, because it is
a dimensionless RATIO to a bare ``GemmSm90`` launch measured back-to-back in the same session, which
is what makes it survive the load and the machine that the absolute milliseconds do not.
"""

import os

import pytest
import torch

from benchmark.distributed import bench_utils as BU
from tests.perf.calibration import assert_cell, assert_host_dispatch
from tests.perf.pins import load as load_pins
from fold_cp_ops.kernels.gemm_hadamard import gemm_hadamard
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from tests.kernels.test_gemm_hadamard import GEMM_HADAMARD, _gen  # the ONE declaration, imported

# ── SM90 gate ──────────────────────────────────────────────────────────────────────────────────
_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(
    _SM != 9, reason=f"pins are SM90 (H100) numbers; this GPU is sm_{_SM}0"
)

_MEASURE = os.environ.get("CPO_PERF_MEASURE", "0") == "1"

_ROUNDS, _WARMUP, _ITERS = 30, 10, 10

#: This gate's pinned numbers, loaded from the sibling JSON of the SAME NAME. The name is the
#: documentation: a number in a failure message is traceable to its file without grepping, and a pin
#: file names the one test that consumes it.
PINS = load_pins(__file__)


def _time(fn, key, flops):
    """Time ``fn``, report TFLOP/s, and gate on the pin for ``key``.

    Args:
        fn: A zero-argument callable issuing exactly one ``gemm_hadamard()`` call. Must already have
            been called once, so the JIT compile is not inside the timed region -- a cold compile
            there is tens of seconds and would dominate every sample.
        key: The cell's identity, used as the key into :data:`PINS` and printed on failure. Must
            match the pin file's ``key`` exactly, element types included; a mismatch reads as
            "unpinned" and SKIPS rather than failing.
        flops: Floating-point operations per call, for the reported TFLOP/s. Informational only --
            the gate is on wall time, never on the derived rate.

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


def _operands(M, N, K, L, dev, bias="row"):
    """Build one cell's operands, already warm. Returns ``(call, flops)``.

    Args:
        M: Rows -- the TriMul token-pair count at the production cells.
        N: Output feature dim.
        K: Contracted feature dim.
        L: Batch count.
        dev: The device fixture's device.
        bias: ``"none"`` or ``"row"``; a column bias would impose a 4-element floor on M, which two
            of the shape cells below do not satisfy.

    Returns:
        ``(call, flops)`` with ``call`` a zero-argument closure already invoked once, so the compile
        is outside the timed region, and ``flops`` the per-call ``2*L*M*N*K``.
    """
    dt = torch.bfloat16
    A, B = _gen((L, M, K), dt), _gen((L, N, K), dt)
    C = _gen((L, M, N), dt)
    D = torch.empty(L, M, N, device=dev, dtype=dt)
    rv = _gen((L, N), torch.float32) if bias == "row" else None
    call = lambda: gemm_hadamard(A, B, D, C, None, 128, 128, 1, 1, rowvec_bias=rv)  # noqa: E731
    call()
    torch.cuda.synchronize()
    return call, 2 * L * M * N * K


# ── the shape grid: where this kernel actually runs ───────────────────────────────────────────
# The gate is a LIST of cells, not the product of the pools: a perf cell pins one number and costs a
# full timed run, so the product would be hundreds of them. Correctness coverage of the full pools
# stays in tests/kernels/test_gemm_hadamard.py.
_SHAPE_CELLS = [
    (4096, 512, 512, 1),  # compute-bound, square-ish
    (1001, 512, 512, 1),  # compute-bound with an M tail
    (4096, 512, 8, 1),  # memory-bound: thin K, so the gate's C read dominates
    (301, 200, 136, 3),  # small and off-grid on both extents: launch/tail-bound
    # --- the A2A-fused TriMul back projection (reference ops 9 + 12) at production size:
    # M = N_token**2 / cp at cp = 16, N = K = D. This is the ONLY caller of this kernel.
    (262144, 128, 128, 1),  # N_token = 2048,  D = 128
    (1048576, 256, 256, 1),  # N_token = 4096,  D = 256
    (1048576, 384, 384, 1),  # N_token = 4096,  D = 384
    (1048576, 512, 512, 1),  # N_token = 4096,  D = 512
]


@GEMM_HADAMARD.parametrize(
    "M",
    "N",
    "K",
    "L",
    cells=_SHAPE_CELLS,
    because=(
        "a perf gate pins ONE number per cell, so the grid is a list and not the 12 x 11 x 10 x 8 "
        "product of the shape pools. The first four span the regimes whose bottleneck differs -- "
        "compute-bound, compute-bound with an M tail, thin-K memory-bound where reading the gate is "
        "most of the traffic, and a small off-grid launch-bound shape -- which a single-size gate "
        "cannot distinguish. The last four are this kernel's only production caller, at one token "
        "count against all four feature dims. N_token = 8192 and 12288 are omitted: they are the "
        "same arithmetic intensity as 4096 at 4x and 9x the runtime and the memory, so they buy "
        "resolution nothing and cost the gate minutes; correctness covers them."
    ),
)
def test_gemm_hadamard_perf_shapes(M, N, K, L, dev):
    """The fused-gate entry holds its pinned median at each shape.

    A row bias is present at every cell because the production caller passes one (``p_out_b``), so
    a cell without it would pin a configuration nothing runs. What varies between cells is the
    shape and nothing else.
    """
    torch.manual_seed(0)
    call, flops = _operands(M, N, K, L, dev)
    _time(call, ("shape", M, N, K, L), flops)


@GEMM_HADAMARD.parametrize(
    "bias",
    only={"bias": ["none", "row", "both"]},
    because=(
        "'col' is omitted because 'both' already carries the column-bias load and the pair "
        "none/row/both is what makes each term's cost readable as a difference. Correctness for "
        "all four is in tests/kernels/test_gemm_hadamard.py."
    ),
)
def test_gemm_hadamard_perf_epilogue_terms(bias, dev):
    """Each additive term's cost is pinned at ONE fixed, compute-bound shape.

    Fixing the shape is what makes these numbers readable: the delta between the ``none`` cell and
    the ``both`` cell is the bias cost, and it is only a bias cost if nothing else moved. The gate
    itself is present in every cell -- it is not optional at this entry -- so its cost is read
    against ``test_benchmark_perf_gemm.py``'s ``('epi', 'none', 'scalar')`` cell, which is the same
    C traffic under an addition instead of a multiply.
    """
    torch.manual_seed(0)
    M, N, K, L = 4096, 2048, 1024, 1
    dt = torch.bfloat16
    A, B = _gen((L, M, K), dt), _gen((L, N, K), dt)
    C = _gen((L, M, N), dt)
    D = torch.empty(L, M, N, device=dev, dtype=dt)
    rv = _gen((L, N), torch.float32) if bias in ("row", "both") else None
    cv = _gen((L, M), torch.float32) if bias in ("col", "both") else None
    call = lambda: gemm_hadamard(  # noqa: E731
        A, B, D, C, None, 128, 128, 1, 1, rowvec_bias=rv, colvec_bias=cv
    )
    call()
    torch.cuda.synchronize()
    _time(call, ("epi", bias), 2 * L * M * N * K)


@matrix_exempt("asserts a property of the two timed tests' SOURCE; there is nothing to sweep")
def test_no_timed_cell_runs_the_autotuner():
    """A pinned cell must measure a KERNEL, never a sweep.

    ``do_autotune`` defaults to False, so this holds by omission -- which is exactly why it is worth
    an assertion: nothing about the call sites says the tuner is off, and switching it on would
    silently replace a 0.05 ms kernel time with a multi-second sweep that still "passes" the first
    time it is harvested.
    """
    import inspect

    for fn in (test_gemm_hadamard_perf_shapes, test_gemm_hadamard_perf_epilogue_terms):
        src = inspect.getsource(fn)
        assert "do_autotune" not in src, (
            f"{fn.__name__} sets do_autotune; a pinned cell must measure a kernel, never a sweep."
        )


@matrix_exempt("gates the host submit path, which is shape-independent; there is nothing to sweep")
def test_the_gemm_hadamard_entry_dispatch_cost_holds(dev):
    """The PER-CALL PYTHON cost of the `gemm_hadamard` entry holds, with the CPU divided out.

    The companion to the timed cells and the half they cannot see: those are timed with
    ``mode="device"``, which reports the kernel and nothing about the submit path. This entry does
    strictly MORE host work than `gemm` -- a fifth `check_tensor` for the required C plus its shape
    comparison against D -- so it is the one cell in this file that is expected to differ from the
    stock GEMM's by a measurable amount, and the one worth pinning before the timed cells are.

    Measured at a TINY shape on purpose: dispatch is shape-independent for a given entry point, so
    this is the same number the big cells pay, obtained without their runtime. It is calibrated
    against a BARE ``GemmSm90`` launch measured back-to-back, so what is gated is this entry's cost
    RELATIVE to a plain CuTe-DSL submit -- which is what survives a change of machine, where the
    absolute microseconds do not.
    """
    m, n, k, ell = 8, 8, 8, 1
    dt = torch.bfloat16
    A, B = _gen((ell, m, k), dt), _gen((ell, n, k), dt)
    C = _gen((ell, m, n), dt)
    D = torch.empty(ell, m, n, device=dev, dtype=dt)
    call = lambda: gemm_hadamard(A, B, D, C, None, 64, 64, 1, 1)  # noqa: E731
    call()
    torch.cuda.synchronize()
    assert_host_dispatch(call, ("dispatch", "gemm_hadamard"), PINS, __file__, device=dev)
