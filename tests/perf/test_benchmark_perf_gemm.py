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

"""PERF GATE: ``fold_cp_ops.kernels.gemm.gemm`` -- the host entry with the default epilogue.

**Do NOT lock the GPU clock for this gate**, and do not harvest its pins with one either.

Locking makes any single measurement far more repeatable, but it SHIFTS THE MEAN: an H100's
free-running clock follows the 700 W power cap and hence the workload, so a locked pin and a
free-running measurement are samples of two different distributions. Mixing them produced 23
failures in one suite, none of them real. Both sides run free-running, and the sample counts
(``calibration.N_SAMPLES_PIN_TIME`` / ``N_SAMPLES_TEST_TIME``) are what make the medians converge --
1/sqrt(N), measured.

Pins a median (ms/call) per cell and fails if the measured median drifts above the band for
that cell's own measured band. The band is derived from the spread pinned alongside the median,
not from a constant: the launch-bound cells' own spread is 5-7% and the compute-bound ones' is
under 1%, so one tolerance could not serve both.

**This gate and ``test_benchmark_perf_gemm_sm90.py`` are a pair, and the pairing is the point.**
That one times the bare mainloop; this one times the same mainloop with the default epilogue's
alpha/beta/bias terms compiled in. Read together, a regression that appears here and not there is in
the epilogue, and one that appears in both is in the mainloop. Either gate alone can only say "the
GEMM got slower".

The epilogue axis is therefore swept *at a fixed shape*: what is being measured is the cost of each
epilogue term, not of the matmul underneath it.

Timing goes through ``bench_utils.benchmark_single(mode="device")`` per CLAUDE.md -- never a
hand-rolled ``perf_counter`` loop around an async launch, which measures host dispatch.

Harvest new pins (prints medians and TFLOP/s, skips the assertion)::

    CPO_PERF_MEASURE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s \
        tests/perf/test_benchmark_perf_gemm.py
"""

import os

import pytest
import torch

from benchmark.distributed import bench_utils as BU
from tests.perf.calibration import assert_cell, assert_host_dispatch, host_dispatch_us
from tests.perf.pins import load as load_pins
from fold_cp_ops.kernels.gemm import gemm
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from tests.kernels.test_gemm import GEMM, _gen  # the ONE declaration, imported

# ── SM90 gate ──────────────────────────────────────────────────────────────────────────────────
_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(
    _SM != 9, reason=f"pins are SM90 (H100) numbers; this GPU is sm_{_SM}0"
)

_MEASURE = os.environ.get("CPO_PERF_MEASURE", "0") == "1"
#: This gate is reference-calibrated: every cell divides its measurement by a reference measured in
#: the SAME session before comparing, so it may run in a shared session. See `calibration`.
#:
#: **This replaced a two-tier tolerance constant, and the replacement is the point.** The old gate
#: chose between a 10% band and a 25% one by whether the pin sat under 0.020 ms -- a hand-drawn
#: threshold standing in for "this cell is launch-bound, so it is noisy". The measured spreads it
#: was approximating range from 0.4% (TriMul front) to 5-7% (dispatch floor), better than a 10x
#: range, and each cell now carries its OWN measured spread in the pin file. The band follows from
#: that number, so nobody picks it and nobody has to revisit the threshold when a kernel gets fast
#: enough to cross it.


_ROUNDS, _WARMUP, _ITERS = 30, 10, 10

#: Pinned medians (ms/call) on an idle H100 80GB HBM3, bf16, k-major operands, persistent
#: non-pingpong grid, tile (128, 128). Keyed by the cell tuple of whichever test pinned it. An
#: absent key measures and skips rather than failing.
#:
#: **Harvested AFTER the front-door work, which the first set was not.** The three launch-bound
#: cells were pinned at 0.0136/0.0133/0.0135 before the base-address and fp32-operand checks
#: existed; those cost ~1 us net even after `describe_operands` fused the lookups, which is ~7% at
#: this floor, so the old pins failed roughly half the time. Re-measuring beat re-tightening: the
#: cost is real, it is bought with three refusals that used to be FFI errors, and a pin that
#: encodes it is more useful than one that argues with it.
#:
#: **The epilogue block is the one to read as a set**, all at M/N/K/L = 4096/2048/1024/1 so the
#: differences are the terms and nothing else: no bias 0.0357 ms, row bias +6%, column bias +4%,
#: both +7%, and a C addend with a scalar beta +24% -- the last being an extra full-size tensor read
#: rather than a broadcast, which is why it costs several times what a bias does.
#:
#: The ``('shape', 4096, 2048, 2048, 1)`` cell at 0.0656 ms sits within 1% of the same shape's
#: 0.0650 ms in ``test_benchmark_perf_gemm_sm90.py`` at the same tile. That agreement is the
#: cross-check the two gates exist to provide: with the epilogue at its cheapest, the default entry
#: should cost what the bare mainloop costs.
#:
#: The three TriMul front cells run at 263-473 TFLOP/s. The D=128 cell is the slow one and it is
#: slow for a structural reason: K = 128 is two 64-wide k-tiles, so the mainloop has almost no
#: steady state to amortize its prologue over. That is what the front projection *is* -- the number
#: to watch is whether it moves, not whether it approaches the back einsum's 640.
#: This gate's pinned numbers, loaded from the sibling JSON of the SAME NAME. The name is the
#: documentation: a number in a failure message is traceable to its file without grepping, and a
#: pin file names the one test that consumes it. Enforced in conftest -- a timed perf module that
#: does not expose `PINS` FAILS rather than skips, because a gate inventing its own number format
#: is exactly what that check exists to catch.
PINS = load_pins(__file__)


def _time(fn, key, flops):
    """Time ``fn``, report TFLOP/s, and gate on the pin for ``key``.

    Args:
        fn: A zero-argument callable issuing exactly one ``gemm()`` call. Must already have been
            called once so the JIT compile is not inside the timed region.
        key: The cell's identity, used as the key into :data:`PINS` and printed on failure.
        flops: Floating-point operations per call, for the reported TFLOP/s. Only informational --
            the gate is on wall time.

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


# ── the shape grid: the three regimes a GEMM moves between ────────────────────────────────────
_SHAPE_CELLS = [
    (4096, 2048, 2048, 1),  # compute-bound
    (1001, 2048, 2048, 1),  # compute-bound with an M tail
    (4096, 2048, 64, 1),  # memory-bound: thin K
    (301, 200, 256, 3),  # small and off-grid: launch/tail-bound
    (128, 256, 256, 7),  # small, batched
    # --- the A2A-fused TriMul FRONT projection, at its production size ---
    # M = N_token^2 flattened tokens, K = D in {128, 256, 384}, N = the stacked gate/up postact
    # width. This half contracts over the FEATURE dim, so it is O(N_token^2) and is explicitly
    # EXEMPT from the K = N_token rule that governs the back einsum.
    (1000000, 512, 128, 1),  # N_token=1000, D=128
    (1000000, 1024, 256, 1),  # N_token=1000, D=256
    (1000000, 768, 384, 1),  # N_token=1000, D=384
]


@GEMM.parametrize(
    "M",
    "N",
    "K",
    "L",
    cells=_SHAPE_CELLS,
    because=(
        "a perf gate pins ONE number per cell, so the grid is a list and not the 7 x 8 x 7 x 4 "
        "product of the shape pools. These five span the regimes whose bottleneck differs -- "
        "compute-bound, compute-bound with an M tail, thin-K memory-bound, and two small "
        "launch-bound shapes -- which a single-size gate cannot distinguish. Correctness coverage "
        "of the full pools stays in tests/kernels/test_gemm.py."
    ),
)
def test_gemm_perf_shapes(M, N, K, L):
    """The default-epilogue entry holds its pinned median at each shape.

    The epilogue is at its cheapest here (no C, no bias, alpha folded away), so what these cells
    watch is the mainloop plus the plain store. ``test_gemm_perf_epilogue_terms`` adds the terms
    back one at a time, at a fixed shape.
    """
    torch.manual_seed(0)
    A, B = _gen((L, M, K), torch.bfloat16), _gen((L, N, K), torch.bfloat16)
    D = torch.empty(L, M, N, device="cuda", dtype=torch.bfloat16)
    call = lambda: gemm(A, B, D, None, None, 128, 128, 1, 1)  # noqa: E731
    call()  # compile outside the timed region
    torch.cuda.synchronize()
    _time(call, ("shape", M, N, K, L), 2 * L * M * N * K)


@GEMM.parametrize(
    "bias",
    "beta_mode",
    only={"beta_mode": ["one", "scalar"]},
    because=(
        "the epilogue-cost sweep needs the C term present or absent, not the three ways of "
        "supplying beta -- a device-pointer beta adds one scalar load per CTA, which is below this "
        "gate's 10% band. The pointer path's correctness is covered in tests/kernels/test_gemm.py."
    ),
)
def test_gemm_perf_epilogue_terms(bias, beta_mode):
    """Each epilogue term's cost is pinned at ONE fixed shape.

    Fixing the shape is what makes these numbers readable: the delta between the no-bias cell and
    the both-bias cell is the bias cost, and it is only a bias cost if nothing else moved. The shape
    chosen is compute-bound, where an epilogue regression is most visible as a fraction of a fast
    kernel rather than lost in a memory-bound one.
    """
    torch.manual_seed(0)
    M, N, K, L = 4096, 2048, 1024, 1
    A, B = _gen((L, M, K), torch.bfloat16), _gen((L, N, K), torch.bfloat16)
    D = torch.empty(L, M, N, device="cuda", dtype=torch.bfloat16)
    C = _gen((L, M, N), torch.bfloat16) if beta_mode != "one" else None
    rv = _gen((L, N), torch.float32) if bias in ("row", "both") else None
    cv = _gen((L, M), torch.float32) if bias in ("col", "both") else None
    beta = -0.5 if beta_mode == "scalar" else 1.0
    call = lambda: gemm(  # noqa: E731
        A, B, D, C, None, 128, 128, 1, 1, rowvec_bias=rv, colvec_bias=cv, beta=beta
    )
    call()
    torch.cuda.synchronize()
    _time(call, ("epi", bias, beta_mode), 2 * L * M * N * K)


# ── the autotuned entry: what it PICKS, and whether the pick is worth the sweep ────────────────
# Three assertions per cell, because they fail for different reasons and one hides the others:
#
#   selection  -- WHICH config won. A regression that changes the pick is invisible to a timing
#                 gate whenever the two configs happen to time alike at that shape, and it is
#                 exactly the regression that will hurt at a shape nobody pinned.
#   payoff     -- the tuned config, called through the PINNED path, must be no worse than the
#                 fixed-config cell pinned above. That is the real bar: autotuning that lands on a
#                 config slower than the hardcoded default is worse than not autotuning, and a pin
#                 on the tuned number alone cannot say that -- it would ratchet down with the pool.
#   dispatch   -- the tuner's WARM per-call cost, pinned separately. Measured on this box: a warm
#                 tuned call costs ~12 us more host time than a pinned one, which is invisible on a
#                 compute-bound shape and roughly DOUBLES a launch-bound one. That is why the
#                 payoff assertion uses the pinned path: mixing the two would report a real
#                 kernel-selection win as a loss, or hide a dispatch regression inside a kernel win.
#
# The dispatch number is a fact about the tuner, not about the GEMM, and it is the reason
# ``_config=`` exists. It was ~20 us before the warm path stopped rebuilding the knob-name set and
# re-running `validity` over the whole pool on every call -- both O(pool) per launch.
#
def _fixed_call(A, B, D, tile_shape_mn, pingpong, cluster_mn):
    """A zero-argument callable running `gemm` DIRECTLY with one config's knobs.

    Purpose
        Both selection assertions compare a config against a config, so both sides must go through
        the same call path -- and that path is the un-wrapped one, because the tuner's dispatch is
        gated separately.

    Args:
        A: `(l, m, k)` activation.
        B: `(l, n, k)` weight.
        D: `(l, m, n)` output, written in place.
        tile_shape_mn: `(tile_M, tile_N)`.
        pingpong: The schedule.
        cluster_mn: `(cluster_M, cluster_N)`.

    Returns:
        A callable issuing exactly one `gemm()` launch. Not yet warmed -- the caller must invoke it
        once before timing so the JIT compile is outside the timed region.
    """
    return lambda: gemm(A, B, D, None, None, *tile_shape_mn, *cluster_mn, pingpong=pingpong)


#: The best config KNOWN for each cell, and the thing the tuner's pick is judged against.
#:
#: **Not the pick itself, and that correction is the content of this gate.** Pinning "the tuner
#: chose X" was tried and is flaky by construction: measured over five consecutive harvest runs on
#: an idle box, two of these three cells chose a different config almost every time -- 4096x2048x64
#: is launch-bound so nearly every config ties, and at 4096x2048x2048 the top three measured 0.0488,
#: 0.0489 and 0.0492 ms, a 0.8% spread below the timer's own noise. Which one wins carries no
#: information, so an equality assertion on it is a coin flip dressed as a gate.
#:
#: What DOES carry information is whether the pick is materially worse than the best available. So
#: the reference config below is re-timed in the same run, under the same conditions, and the
#: winner must come within `_TIE_BAND` of it. That fails when the tuner starts choosing badly and
#: passes when it chooses any of several equivalent configs -- which is exactly the distinction the
#: pinned-pick version could not make.
_AUTOTUNE_REFERENCE = {
    (4096, 2048, 2048, 1): ((128, 256), False, (1, 2)),
    (1001, 2048, 2048, 1): ((128, 128), True, (2, 1)),
    (4096, 2048, 64, 1): ((128, 256), False, (1, 2)),
}

#: How far above the reference config a pick may measure and still count as equivalent. 10% is well
#: above the 0.8-2% spread among tied configs and well below any real selection regression.
_TIE_BAND = 1.10

#: The fixed-config cell each autotuned cell must beat: same shape, same epilogue, tile (128, 128)
#: cluster (1, 1). So the comparison is exactly "did sweeping 22 configs help".
_AUTOTUNE_BASELINE_KEY = {
    (4096, 2048, 2048, 1): ("shape", 4096, 2048, 2048, 1),
    (1001, 2048, 2048, 1): ("shape", 1001, 2048, 2048, 1),
    (4096, 2048, 64, 1): ("shape", 4096, 2048, 64, 1),
}

#: **The payoff is measured on a DIRECT call to the winning config, not through the wrapper**, and
#: the reason is a decomposition worth keeping. At 4096x2048x64 -- launch-bound, ~14.5 us -- the
#: three numbers are:
#:
#:   fixed cell,  gemm(128, 128, 1, 1) direct   0.0145 ms
#:   the WINNER,  gemm(*winner)        direct   0.0149 ms   +2.9%  <- the config choice
#:   the WINNER,  through `_config=`            0.0168 ms   +16%   <- +1.9 us of wrapper
#:
#: So a payoff assertion timed through the wrapper reports a 1.16x "regression" that is ~87% Python
#: dispatch and ~13% config choice. At this floor even the pinned path's ~2 us is 13% of the kernel.
#: Timing the winner directly asks the question the pool is responsible for -- "did sweeping find a
#: good config" -- and leaves the wrapper's cost to the dispatch assertion below, where a change in
#: it is legible as a change in the TUNER.
#:
#: **A correction worth recording: this gap is NOT the absence of cluster (1, 1) in the pool.** That
#: was the first explanation and it was wrong. Measured across five shapes, the pool's best and the
#: best over the pool PLUS all eleven (1, 1) variants are the same to within noise -- (1, 1) ties at
#: the thin-K shape (0.0145 vs 0.0145) and LOSES everywhere else, by 1% compute-bound and 11% at the
#: M-tail shape. The upstream project drops (1, 1) deliberately (2025-09-07, "[Gemm] Only tune
#: cluster 2x1 and 1x2, don't tune 1x1"), and the measurement supports it: adding it grows every
#: sweep by 50% for no win. The pool can reach 0.0145 ms here; the tuner's pick is 0.0149.
#:
#: Payoff, measured on the direct call:
#:
#:   4096x2048x2048   1.36x win (703 TFLOP/s)
#:   1001x2048x2048   1.20x win
#:   4096x2048x64     within 3% of the fixed cell -- parity, and the pool contains no better
#:
#: Ceiling on the tuner's WARM per-call host overhead, in microseconds, over the same kernel called
#: with `do_autotune` off at the same shape. Measured ~8 us on this box once the two entry points
#: collapsed into one gated function (it was 12-14 us through the extra wrapper layer, and ~20 us
#: before the O(pool) work left the warm path). 25 leaves room for run-to-run spread on a
#: Python-level measurement while still catching a per-call O(pool) walk creeping back in.
_DISPATCH_OVERHEAD_CEILING_US = 25.0


@matrix_exempt(
    "sweeps the AUTOTUNER over a shape list; the kernel matrix declares correctness cells, and "
    "one autotune cell costs 22 compiles plus 22 timed sweeps"
)
@pytest.mark.parametrize("M,N,K,L", list(_AUTOTUNE_BASELINE_KEY))
def test_gemm_autotune_picks_a_good_config_and_beats_the_fixed_default(M, N, K, L):
    """Three separate things, because they fail for different reasons and one would hide the others.

    **Selection** -- the pick must come within `_TIE_BAND` of the best KNOWN config for this shape,
    re-timed in the same run. Not "the pick equals X": measured over five harvest runs, two of these
    three cells chose a different config almost every time, because the top configs tie within the
    timer's noise. An equality assertion there is a coin flip; this one fails only when the tuner
    starts choosing materially worse, which is the thing worth catching.

    **Payoff** -- the tuned config must be no worse than the FIXED-config cell pinned above at the
    same shape and epilogue. A comparison rather than a pin, because a pinned tuned median would
    ratchet: widen the pool, measure, re-pin, and the gate could never say the sweep stopped paying.

    **Dispatch** -- the tuner's warm per-call host cost, gated separately, and kept OUT of the two
    timing assertions above. Both of those time a DIRECT ``gemm()`` call with the winning knobs, so
    what they measure is the config the sweep chose and nothing else. Measured at the launch-bound
    cell: the winner direct is 0.0149 ms and the same winner through ``_config=`` is 0.0168 -- a
    1.16x "regression" that is ~87% Python dispatch. Mixing the two would report a dispatch cost as
    a bad pick, or hide a dispatch regression inside a kernel win.

    Args:
        M: Rows of A and D.
        N: Columns of D and rows of B.
        K: Contraction extent.
        L: Batch.
    """
    torch.manual_seed(0)
    A, B = _gen((L, M, K), torch.bfloat16), _gen((L, N, K), torch.bfloat16)
    D = torch.empty(L, M, N, device="cuda", dtype=torch.bfloat16)
    gemm(A, B, D, None, None, do_autotune=True)  # tunes, compiles and caches the winner
    torch.cuda.synchronize()

    won = gemm.autotuner.best_config
    pick = ((won["tile_M"], won["tile_N"]), won["pingpong"], (won["cluster_M"], won["cluster_N"]))

    direct = _fixed_call(A, B, D, *pick)
    warm = lambda: gemm(A, B, D, None, None, do_autotune=True)  # noqa: E731
    direct()
    warm()
    torch.cuda.synchronize()
    median_ms = BU.benchmark_single(
        direct, mode="device", rounds=_ROUNDS, warmup=_WARMUP, iters=_ITERS
    ).median_ms
    tflops = 2 * L * M * N * K / (median_ms * 1e-3) / 1e12

    # Host dispatch, deliberately: this is a Python cost, and the launches are async, so a host-side
    # span around them measures exactly the thing being gated. It is NOT a device timing.
    #
    # Through `host_dispatch_us` rather than a local loop, and the local loop it replaces was WRONG
    # in a way that silently disabled this assertion at one of the three cells. It read:
    #
    #     t0 = perf_counter(); for _ in range(n): fn(); synchronize(); return (perf_counter()-t0)/n
    #
    # -- the trailing drain is INSIDE the span, so the elapsed contains all n kernels' DEVICE time.
    # Where the kernel is slower than the dispatch, the reading is device time wearing dispatch's
    # clothes, and the difference of two such readings collapses. Measured on an H200, before/after:
    #
    #     cell                    device      was      now
    #     (4096,2048,2048)      0.0491 ms   +0.0 us   +9.6 us     <- the gate was dead here
    #     (1001,2048,2048)      0.0182 ms   +8.2 us   +9.7 us
    #     (4096,2048,  64)      0.0112 ms   +8.5 us  +10.3 us
    #
    # The two small cells move too, and upward: their drain also diluted the difference, just less.
    # The three now agree to 0.7 us, which is what a per-call host cost that does not depend on the
    # kernel should look like -- the old spread of 0.0-8.5 was the kernels' device times leaking in.
    # The test also got 4.5x faster (70.7 s -> 15.5 s): the old loop waited out 500 kernels twice.
    #
    # The assertion below is ONE-SIDED (`<= ceiling`), so a collapsed reading passes unconditionally
    # -- the O(pool) walk this gate exists to catch could come back at that cell and nothing would
    # say so. `host_dispatch_us` captures `elapsed` BEFORE its drain, which is the whole difference,
    # and adds the dual-loop-count backpressure check the local copy had no equivalent of.
    overhead_us = host_dispatch_us(warm) - host_dispatch_us(direct)

    cell = (M, N, K, L)
    if _MEASURE or cell not in _AUTOTUNE_REFERENCE:
        print(
            f"    autotune {cell}: {pick}  {median_ms:.4f} ms  ({tflops:6.1f} TFLOP/s)"
            f"  warm-dispatch +{overhead_us:.1f} us"
        )
        if not _MEASURE:
            pytest.skip(f"no autotune pin for {cell}; run with CPO_PERF_MEASURE=1 to harvest one")

    # Selection: judged against the best KNOWN config, re-timed here so the comparison is between
    # two numbers from the same run rather than against a stale pin. Both sides are DIRECT calls,
    # so neither carries the wrapper -- see the note on `_AUTOTUNE_REFERENCE`.
    ref_call = _fixed_call(A, B, D, *_AUTOTUNE_REFERENCE[cell])
    ref_call()
    torch.cuda.synchronize()
    ref_ms = BU.benchmark_single(
        ref_call, mode="device", rounds=_ROUNDS, warmup=_WARMUP, iters=_ITERS
    ).median_ms
    assert median_ms <= ref_ms * _TIE_BAND, (
        f"autotune {cell} picked {pick} at {median_ms:.4f} ms, against the best known config "
        f"{_AUTOTUNE_REFERENCE[cell]} at {ref_ms:.4f} ms -- {median_ms / ref_ms:.2f}x, "
        f"outside the {_TIE_BAND:.2f}x band. The tuner is choosing materially worse than it could."
    )
    # Gated on the FIXED cell's pin, so `emit=False`: harvesting here would overwrite that cell's
    # pin with the TUNED number, after which the tuner is compared against its own previous output.
    assert_cell(
        lambda: median_ms,
        _AUTOTUNE_BASELINE_KEY[cell],
        PINS,
        __file__,
        emit=False,
        describe=lambda _: (
            f"autotune {cell} at {tflops:.1f} TFLOP/s vs the FIXED tile (128,128) cell. A sweep "
            f"that lands on a config slower than the hardcoded default is worse than not sweeping."
        ),
    )
    assert overhead_us <= _DISPATCH_OVERHEAD_CEILING_US, (
        f"autotune {cell}: a warm tuned call costs {overhead_us:.1f} us more host time than a "
        f"pinned one, over the {_DISPATCH_OVERHEAD_CEILING_US} us ceiling. Something O(pool) has "
        f"crept back onto the per-call path -- the warm path must be one key build and one dict hit."
    )


@matrix_exempt("a property of this module's own call sites; nothing is launched")
def test_the_fixed_cells_do_not_route_through_the_tuner():
    """Every pinned fixed-config cell calls `gemm` directly, so its number is a kernel's.

    A pinned cell that reached the tuner would time a 22-config sweep on its first call and the
    winner thereafter -- two different numbers from one assertion, depending on ordering. Keeping
    the two entries separate is what makes `CPO_AUTOTUNE` irrelevant to these gates rather than
    something they have to set.
    """
    import inspect

    for fn in (test_gemm_perf_shapes, test_gemm_perf_epilogue_terms):
        src = inspect.getsource(fn)
        assert "do_autotune" not in src, (
            f"{fn.__name__} sets do_autotune; a pinned cell must measure a kernel, never a sweep. "
            f"The gate defaults to off, so omitting it is all that is required."
        )


@matrix_exempt("gates the host submit path, which is shape-independent; there is nothing to sweep")
def test_the_gemm_entry_dispatch_cost_holds(dev):
    """The PER-CALL PYTHON cost of the `gemm` entry holds, with the CPU divided out.

    Purpose
        The companion to the timed cells, and the half they can no longer see. Since the cells moved
        to ``mode="device"`` they report the kernel and nothing else -- which is what makes them
        able to catch a mainloop regression at a small shape for the first time, and also what makes
        them blind to the submit path. This is where that cost is watched.

    Semantics
        Measured at a TINY shape on purpose: dispatch is shape-independent for a given entry point
        (~7% across a 65x range of device times), so this is the same number the big cells pay,
        obtained without their runtime. It is calibrated against a BARE ``GemmSm90`` launch, so what
        is gated is `gemm`'s cost RELATIVE to a plain CuTe-DSL submit -- measured 1.22x, i.e. ~3.8 us
        of epilogue args, scheduler args, ``data_ptr()``, the ``jit_cache`` lookup and the autotune
        gate. Dividing by a same-class reference is what makes that survive a change of machine,
        where the absolute microseconds do not.
    """
    m, n, k, ell = 8, 8, 8, 1
    A, B = _gen((ell, m, k), torch.bfloat16), _gen((ell, n, k), torch.bfloat16)
    D = torch.empty(ell, m, n, device=dev, dtype=torch.bfloat16)
    call = lambda: gemm(A, B, D, None, None, 64, 64, 1, 1)  # noqa: E731
    call()
    torch.cuda.synchronize()
    assert_host_dispatch(call, ("dispatch", "gemm"), PINS, __file__, device=dev)
