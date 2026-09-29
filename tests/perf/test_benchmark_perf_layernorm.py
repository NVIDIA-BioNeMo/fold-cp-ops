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

"""PERF GATE: ``fold_cp_ops.kernels.layernorm.layernorm_fwd`` on a single SM90 GPU.

**Do NOT lock the GPU clock for this gate**, and do not harvest its pins with one either.

Locking makes any single measurement far more repeatable, but it SHIFTS THE MEAN: an H100's
free-running clock follows the 700 W power cap and hence the workload, so a locked pin and a
free-running measurement are samples of two different distributions. Mixing them produced 23
failures in one suite, none of them real. Both sides run free-running, and the sample counts
(``calibration.N_SAMPLES_PIN_TIME`` / ``N_SAMPLES_TEST_TIME``) are what make the medians converge --
1/sqrt(N), measured.

Pins a median (ms/call) per ``(M, N, dtype)`` cell and fails if the measured median drifts above
its own measured band of it -- a regression gate, not a benchmark report.

**The grid is what the kernel actually sees: ``(M rows, N features, dtype)``.** Not the TriMul e2e
triple ``(D, N_token, cp)`` -- LayerNorm is a single-GPU row-wise reduction and cannot observe
``cp`` at all, so keying on it would couple this gate to the distributed grid while measuring
nothing extra. The production row counts that triple implies are covered directly, as ``M``.

Timing goes through the shared CUDA-event harness (``bench_utils.benchmark_single(mode="device")``)
per CLAUDE.md -- never a hand-rolled ``perf_counter`` loop, which would measure host dispatch
rather than device time. Clocks are locked for the session by the autouse ``clock_locked``
fixture, which warns and proceeds unlocked where clock control is denied.

Harvest new pins (prints medians, skips the assertion)::

    CPO_PERF_MEASURE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s \
        tests/perf/test_benchmark_perf_layernorm.py

then merge the printed ``PINHARVEST`` lines into the sibling JSON with ``pins.merge_harvest``.
"""

from __future__ import annotations


import pytest
import torch

from benchmark.distributed import bench_utils as BU
from tests.perf.calibration import assert_cell, assert_host_dispatch
from tests.perf.pins import load as load_pins
from fold_cp_ops.kernels.layernorm import layernorm_fwd
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from tests.kernels.test_layernorm import LAYERNORM  # the ONE declaration, imported

# ── SM90 gate ──────────────────────────────────────────────────────────────────────────────────
_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(
    _SM != 9, reason=f"pins are SM90 (H100) numbers; this GPU is sm_{_SM}0"
)

#: This gate is reference-calibrated, so it may run in a shared session. It replaced a flat 10%
#: band applied to all 37 cells, whose own spreads differ by more than 10x across the grid: the
#: 256-row cells sit at the launch floor and are the noisy ones, while the large-N cells are
#: bandwidth-bound and very stable. One constant was simultaneously too loose for the second group
#: and close to a coin flip for the first; each cell's pinned spread now sets its own band.
_ROUNDS, _WARMUP, _ITERS = 50, 10, 10

# ── the grid: what the kernel actually sees ────────────────────────────────────────────────────
# Three things this grid is built to catch, none of which a power-of-two ladder would:
#
# 1. **The production shapes.** The TriMul front calls this kernel with feature dim D in
#    {128, 256, 384} and a row count M = (N_token//cp) * N_token, which spans 5e4 .. 2.5e7 in the
#    e2e grid. A gate that stops at M=65536 never sees the regime the workflow actually runs in,
#    where the kernel is at ~3000 GB/s and any regression is pure lost throughput.
# 2. **Off-grid N.** `predicate_k` AND the masked-tail fill in the variance pass are compiled in
#    ONLY when N is not a tile multiple, so an aligned-only grid leaves their cost unmeasured.
#    The off-grid cells here span every downstream path: plain (760/1000/1128/3000),
#    `reload_from="smem"` (20000), a thread-block cluster (40000), vecsize 2 (1002), and odd N
#    where vecsize collapses to 1 (999/4095/8191 -- float32 only; the 16-bit copy atom an odd N
#    would need at bf16 fails IR verification, see tests/kernels/test_layernorm.py).
# 3. **Both sides of every `delay_w_load` boundary.** The gate's thresholds were fitted to measured
#    cells, so each boundary is bracketed by the last healthy shape and the first collapsed one.
#    Cells marked "pins the threshold from above" are the ones that REGRESS (13-19%) if the gate is
#    made more eager -- they are why the table is keyed on the rung and not just the dtype.
_CELLS = [
    # --- production: the TriMul front's feature dims at production row counts ---
    (1048576, 128, torch.bfloat16),
    (1048576, 256, torch.bfloat16),
    (1048576, 384, torch.bfloat16),
    (262144, 384, torch.bfloat16),
    # --- the _threads_per_row ladder, small end ---
    (65536, 64, torch.bfloat16),  # tpr rung 8 -- the smallest, 16 rows per CTA tile
    (65536, 128, torch.bfloat16),
    (65536, 256, torch.bfloat16),
    (65536, 384, torch.bfloat16),
    (65536, 760, torch.bfloat16),  # off-grid N
    (65536, 1000, torch.bfloat16),  # off-grid N
    (65536, 1002, torch.bfloat16),  # off-grid N with vecsize 2
    (65536, 1024, torch.bfloat16),
    (65536, 1128, torch.bfloat16),  # off-grid N
    (16384, 3000, torch.bfloat16),  # off-grid N
    (16384, 4096, torch.bfloat16),
    # --- the register-spill cliff at threads_per_row = 32, bracketed ---
    (4096, 2688, torch.bfloat16),  # last healthy: est 252, gate must NOT fire
    (4096, 2816, torch.bfloat16),  # first collapse: est 264, gate fires -> 11.1x
    (4096, 3072, torch.bfloat16),
    (4096, 2688, torch.float32),  # last healthy: est 336, gate must NOT fire
    (4096, 2816, torch.float32),  # first collapse: est 352, gate fires -> 6.2x
    (4096, 3072, torch.float32),
    # --- the same estimate one rung up, where firing would COST: pins the threshold from above ---
    (4096, 6136, torch.float32),  # est 384 at tpr=64; firing here costs 19%
    (4096, 8200, torch.float32),  # est 272 at tpr=128; the bf16 bound here costs 13%
    (4096, 14336, torch.float32),  # est 448 at tpr=128; firing here costs 15%
    # --- the register-spill cliff at threads_per_row = 128 ---
    (4096, 8192, torch.bfloat16),  # last healthy N at threads_per_row=128
    (4096, 12288, torch.bfloat16),  # gate fires -> ~15x
    (4096, 16384, torch.bfloat16),  # worst cell; top of the tpr=128 band
    (4096, 16640, torch.bfloat16),  # first N past the ladder switch -- gate must NOT fire
    (4096, 20000, torch.bfloat16),  # off-grid N, reload_from = "smem"
    (4096, 32768, torch.bfloat16),  # cluster_n = 2
    (4096, 40000, torch.bfloat16),  # off-grid N, cluster_n > 1
    (65536, 256, torch.float32),
    (65536, 1024, torch.float32),
    (16384, 999, torch.float32),  # ODD N -> vecsize 1, scalar copies
    (16384, 4095, torch.float32),  # ODD N, large
    (16384, 8191, torch.float32),  # ODD N and PRIME: tile 8192, so exactly ONE padded column
    (4096, 16384, torch.float32),
]

# H100 80GB HBM3 (sm_90) medians in ms/call, event mode. Harvested on THIS node
# (4x H100 80GB HBM3, NVSwitch, torch 2.11.0+cu130), CUDA_VISIBLE_DEVICES=0, within-run
# std < 0.5% -- so the +/-10% band is comfortably wider than the noise.
# Absent key => unpinned; the cell reports its median and skips.
#
# The four cells that previously recorded the register-spill cliff are now FIXED and re-pinned
# DOWNWARD (never relaxed) -- see docs/refactor_and_fix.md §3.2:
#
#     cell                   before      after     speedup
#     N=12288 bf16          1.2207 ms   0.0803 ms   15.2x
#     N=16384 bf16          1.7791 ms   0.1102 ms   16.1x
#     N=16384 fp32          1.7014 ms   0.2000 ms    8.5x
#     (N=8192 bf16 control)  0.0565 ms   0.0562 ms   1.01x   <- unchanged, as required
#
# Root cause was register spilling to local memory, confirmed by NCU (69.1M local-load +
# 69.0M local-store sectors at N=16384 bf16, 3.89 GB DRAM traffic vs an ideal 268 MB). The fix is
# the register-pressure gate on `delay_w_load` in kernels/layernorm.py; it is bit-identical in
# output (pinned by tests/kernels/test_layernorm.py::test_delay_w_load_is_bit_identical).
#
# The OFF-GRID pins additionally hold the line on the masked-tail fill that the variance pass
# needs (docs/refactor_and_fix.md §3.3). That fill is `const_expr`-pruned on every aligned shape,
# so the aligned cells here double as the proof it costs nothing where it is not needed -- paired
# A/B over this whole grid moved no aligned cell more than 0.4%, and the worst off-grid cost was
# +1.3% (N=40000 bf16). Anything beyond that on an off-grid cell is a real regression.
#: This gate's pinned numbers, loaded from the sibling JSON of the SAME NAME. The name is the
#: documentation: a number in a failure message is traceable to its file without grepping, and a
#: pin file names the one test that consumes it. Enforced in conftest -- a timed perf module that
#: does not expose `PINS` FAILS rather than skips, because a gate inventing its own number format
#: is exactly what that check exists to catch.
PINS = load_pins(__file__)


def _cell_id(M: int, N: int, dt: torch.dtype) -> str:
    """Stable pytest id / pin key suffix for a cell.

    Args:
        M: Row count. Any positive int.
        N: Feature extent. Any positive int meeting the 16-byte floor.
        dt: Element dtype; only its short name is used, so ``torch.`` is stripped.

    Returns:
        A string like ``M65536xN384-bfloat16``, safe for a pytest node id (no spaces or dots).
    """
    return f"M{M}xN{N}-{str(dt).replace('torch.', '')}"


def _bytes_moved(M: int, N: int, dt: torch.dtype) -> int:
    """Minimum DRAM traffic for one LayerNorm call, in bytes: read x once, write y once.

    LayerNorm is memory-bound, so this is the quantity a median should be judged against -- it
    turns an opaque ms number into an achieved-bandwidth figure comparable to the HBM3 roofline.
    The fp32 weight/bias vectors (``2 * N * 4`` bytes) are omitted deliberately: they are O(N)
    against O(M*N) of data and are L2-resident across the grid.

    Args:
        M: Row count.
        N: Feature extent.
        dt: Element dtype of the input/output; sets the per-element size.

    Returns:
        Bytes moved, counting the input read and the output write only. A LOWER bound -- a kernel
        that re-reads x from SMEM/GMEM (which this one does above N=16384, via ``reload_from``)
        moves more, so the derived bandwidth is a floor, not a claim of achieved SoL.
    """
    return 2 * M * N * torch.empty(0, dtype=dt).element_size()


def _make_fn(M: int, N: int, dt: torch.dtype, device: torch.device):
    """Build a warmed zero-arg launch closure for one cell.

    Setup (allocation, weight/bias construction) is folded into the closure's captured state
    rather than repeated per call, and the JIT compile is warmed before returning, so the timed
    region is the kernel launch alone.

    Args:
        M: Row count; must be positive.
        N: Feature extent; must meet the repo's 16-byte alignment floor.
        dt: Input dtype -- one of float16 / bfloat16 / float32 (what ``layernorm_fwd`` accepts).
            Weight and bias are always float32, which the kernel requires.
        device: CUDA device to allocate on. Must be the same device the timing runs on, or the
            measurement times a cross-device launch.

    Returns:
        A zero-argument callable running one ``layernorm_fwd``, already warmed and synchronized.
    """
    x = torch.randn(M, N, device=device, dtype=dt)
    w = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)

    def run():
        layernorm_fwd(x, w, b, eps=1e-6)

    for _ in range(10):  # warm the JIT compile and settle clocks
        run()
    torch.cuda.synchronize(device)
    return run


@LAYERNORM.parametrize(
    "M",
    "N",
    "input_dtype",
    cells=_CELLS,
    because=(
        "a perf gate pins one median per cell, so this grid is a LIST of cells rather than a "
        "cross product -- the full matrix would be hundreds of timed runs. Every component is "
        "still checked against the declared pool, so the shapes that are timed and the shapes "
        "that are correctness-tested cannot drift apart."
    ),
)
def test_layernorm_fwd_perf(dev, M, N, input_dtype):
    """The measured median stays within this cell's own band of its pinned H100 median.

    Args:
        dev: Session device fixture; pinned to this rank's GPU (see conftest).
        M: Row count for this cell.
        N: Feature extent for this cell.
        input_dtype: Element dtype for this cell.

    Raises:
        AssertionError: If the drift-corrected median exceeds the pin by more than its band.
        pytest.skip.Exception: If the cell is unpinned; the median is printed so it can be
            harvested.
    """
    fn = _make_fn(M, N, input_dtype, dev)
    measure = lambda: (  # noqa: E731
        BU.benchmark_single(
            fn,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            iters=_ITERS,
            device=dev,
            mode="device",
            label=f"layernorm M={M} N={N} {input_dtype}",
        ).median_ms
    )
    moved = _bytes_moved(M, N, input_dtype)
    assert_cell(
        measure,
        (M, N, str(input_dtype).replace("torch.", "")),
        PINS,
        __file__,
        describe=lambda ms: f"{_cell_id(M, N, input_dtype)} >={moved / (ms * 1e-3) / 1e9:.0f} GB/s",
    )


@matrix_exempt("gates the host submit path, which is shape-independent; nothing to sweep")
def test_the_layernorm_entry_dispatch_cost_holds(dev):
    """The PER-CALL PYTHON cost of the `layernorm_fwd` entry holds, with the CPU divided out.

    Purpose
        Since the timed cells moved to ``mode="device"`` they report the kernel alone. LayerNorm is
        the cheapest kernel of the three gated entries, so at small M its submit path is the larger
        share of wall time -- which is exactly why it needs watching separately rather than being
        folded into a cell median that no longer contains it.

    Semantics
        A TINY shape, because dispatch is shape-independent per entry point. Calibrated against the
        bare ``GemmSm90`` launch: a GEMM rather than a LayerNorm because what the reference must
        share with the target is the DISPATCH class (CuTe-DSL / TVM-FFI submit), not the kernel's
        arithmetic -- the device side is divided out by construction here.
    """
    call = _make_fn(64, 128, torch.bfloat16, dev)
    torch.cuda.synchronize()
    assert_host_dispatch(call, ("dispatch", "layernorm_fwd"), PINS, __file__, device=dev)
