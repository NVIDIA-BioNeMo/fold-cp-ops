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

"""Parametrized pytest for ``fold_cp_ops/_internal/bench_timing.py``.

Subsumes the three ad-hoc validators (kept faithful, migrated to real assertions):

  * ``benchmark/distributed/bench_utils_device_mode_validate.py`` -> group A + B
    (device-mode == do_bench; the event iters=1 host floor sits above device; larger
    iters amortizes toward device).
  * ``benchmark/bench_harness_validation.py`` -> group C (bench_timing event window ==
    the hand-rolled paired-median event idiom) + group A's do_bench cross-check.
  * ``benchmark/distributed/bench_utils_method_equiv.py`` -> group C (method
    equivalence at device-bound cells) + group D (paired-ratio == single ratio).

``triton.testing.do_bench`` STAYS here as the reference *oracle*. In Phase 2 it is
removed from ``bench_timing``' production path; group A then proves the native
device-mode still matches the oracle within tolerance (the key regression guard).

Tolerances: do_bench and the event window both jitter run-to-run, so we assert
device-vs-do_bench within <=10% (the validators used 8%; 10% for CI robustness),
event-vs-idiom within <=8%, and DIRECTION/ORDERING where an absolute match is
fragile. Group G is torchrun-collected and ``pytest.skip``s when world_size==1.

Run locally (groups A-F):
    CPO_CACHE_ENABLED=0 pytest -q tests/_internal/test_bench_timing.py
Run distributed (group G):
    CUDA_VISIBLE_DEVICES=0,1 CPO_CACHE_ENABLED=0 NVSHMEM_DISABLE_NVLS=1 \
    torchrun --nproc_per_node=2 -m pytest -q \
        tests/_internal/test_bench_timing.py -k distributed
"""

from __future__ import annotations

import math
import statistics

import pytest
import torch

from fold_cp_ops._internal import bench_timing as BU

try:
    from triton.testing import do_bench

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False

# The whole file targets a CUDA box (the distributed group additionally needs
# torchrun). A single CPU-only error test is the sole exception and stays valid.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="bench_timing tests require CUDA"
)

_needs_triton = pytest.mark.skipif(
    not _HAVE_TRITON, reason="triton.testing.do_bench oracle unavailable"
)


# ─────────────────────────────────────────────────────────────────────────────
# Zero-arg launch closures (setup folded in — bench discipline (a)). Each factory
# builds inputs, warm-runs, and returns a closure that is nothing but the launch.
# ─────────────────────────────────────────────────────────────────────────────
def _warm(fn, k: int = 10) -> None:
    for _ in range(k):
        fn()
    torch.cuda.synchronize()


def _gemm_fn(n: int, *, dtype=torch.float16, device="cuda"):
    """One square (n x n) @ (n x n) GEMM into a preallocated out (compute-bound-ish)."""
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    out = torch.empty(n, n, device=device, dtype=dtype)

    def run():
        torch.mm(a, b, out=out)

    _warm(run)
    return run


def _elementwise_fn(n: int, *, dtype=torch.float32, device="cuda"):
    """Elementwise add into a preallocated out (memory-bound)."""
    x = torch.randn(n, n, device=device, dtype=dtype)
    y = torch.randn(n, n, device=device, dtype=dtype)
    out = torch.empty(n, n, device=device, dtype=dtype)

    def run():
        torch.add(x, y, out=out)

    _warm(run)
    return run


def _layernorm_fn(m: int, n: int, *, dtype=torch.float32, device="cuda"):
    """An lnt-like row reduction: F.layer_norm over the last dim (memory-bound)."""
    x = torch.randn(m, n, device=device, dtype=dtype)
    w = torch.randn(n, device=device, dtype=dtype)
    b = torch.randn(n, device=device, dtype=dtype)

    def run():
        torch.nn.functional.layer_norm(x, (n,), w, b)

    _warm(run)
    return run


# Hand-rolled reference paired-median event idiom (verbatim from
# ``benchmark/bench_harness_validation.py``: our_paired_median) — the group-C oracle.
def _time_callable_paired(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _our_paired_median(fn, *, rounds: int = 35, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        _time_callable_paired(fn, iters)
    raw = [_time_callable_paired(fn, iters) for _ in range(rounds)]
    return statistics.median(raw)


@pytest.fixture(scope="module")
def dev():
    return torch.device("cuda", torch.cuda.current_device())


# ─────────────────────────────────────────────────────────────────────────────
# A. device-mode vs do_bench oracle (the deliverable + the Phase-2 native guard).
# ─────────────────────────────────────────────────────────────────────────────
@_needs_triton
@pytest.mark.parametrize("n", [256, 512, 1024])
def test_device_mode_matches_do_bench(dev, n):
    """benchmark_single(mode="device").median_ms within 10% of do_bench median."""
    fn = _gemm_fn(n)
    ref = float(do_bench(fn, return_mode="median"))
    got = BU.benchmark_single(fn, rounds=30, warmup=5, device=dev, mode="device").median_ms
    rel = abs(got - ref) / ref
    assert rel <= 0.10, (
        f"n={n}: device-mode {got:.5f}ms vs do_bench {ref:.5f}ms rel={rel:.2%} > 10%"
    )


@_needs_triton
@pytest.mark.parametrize("shape", [(4096, 4096), (8192, 2048)])
def test_device_mode_matches_do_bench_membound(dev, shape):
    """Memory-bound (layer_norm) device-mode within 10% of do_bench.

    This is the load-bearing guard for the Phase-2 native L2-cache flush: without
    the flush a memory-bound kernel reuses L2 across replicas and reports FASTER
    than do_bench, so this cell would blow past 10%.
    """
    m, n = shape
    fn = _layernorm_fn(m, n)
    ref = float(do_bench(fn, return_mode="median"))
    got = BU.benchmark_single(fn, rounds=30, warmup=5, device=dev, mode="device").median_ms
    rel = abs(got - ref) / ref
    assert rel <= 0.10, (
        f"{shape}: device-mode {got:.5f}ms vs do_bench {ref:.5f}ms rel={rel:.2%} > 10%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# B. host-floor behavior — the event path pays a per-call host-submit tax.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [512, 1024])
def test_event_iters1_ge_device(dev, n):
    """event iters=1 (host-submit + device) sits at/above pure device time."""
    fn = _gemm_fn(n)
    dev_ms = BU.benchmark_single(fn, rounds=30, warmup=5, device=dev, mode="device").median_ms
    ev1_ms = BU.benchmark_single(fn, rounds=30, warmup=5, iters=1, device=dev).median_ms
    # The host floor is a positive OFFSET (its magnitude is machine-specific), so
    # assert direction with a small slack for jitter at device-bound cells.
    assert ev1_ms >= dev_ms * 0.9, f"n={n}: event i=1 {ev1_ms:.5f} should be >= device {dev_ms:.5f}"


@pytest.mark.parametrize("n", [256, 512])
def test_event_large_iters_trends_toward_device(dev, n):
    """event iters=50 median <= iters=1 median: larger iters amortizes host dispatch."""
    fn = _gemm_fn(n)
    ev1 = BU.benchmark_single(fn, rounds=30, warmup=5, iters=1, device=dev).median_ms
    ev50 = BU.benchmark_single(fn, rounds=30, warmup=5, iters=50, device=dev).median_ms
    assert ev50 <= ev1 * 1.05, (
        f"n={n}: event i=50 {ev50:.5f} should be <= i=1 {ev1:.5f} (amortizes)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# C. method equivalence — bench_timing event window == hand-rolled paired-median.
# ─────────────────────────────────────────────────────────────────────────────
#: How many medians of EACH idiom to take before comparing them.
#:
#: This test used to compare ONE median per idiom against an 8% bar, and failed roughly one run in
#: six. That was not a timer defect and not bad luck -- the bar was unreachable. Measured at n=1024,
#: 20 trials:
#:
#:     single-median |rel| :  median 1.88%   p90 7.49%   max 11.77%   over 8% in 2/20
#:     each idiom's own single-median spread : ~12.7%
#:
#: One draw against one draw cannot hold 8% when each draw varies by 12%. Taking the median of
#: several draws each collapses it, but the two sizes need different amounts, so the count is set
#: by the harder one. Median-of-5, 8 trials per size:
#:
#:     n=1024   median 2.68%   max 4.91%   over 8%: 0/8
#:     n=2048   median 0.70%   max 8.22%   over 8%: 1/8
#:
#: n=2048 is usually an order of magnitude TIGHTER than n=1024 (0.3-0.9%) but has a heavy tail:
#: rare excursions of 6-8% that span several consecutive draws, so a median of 5 can still be
#: dragged by them. 9 is chosen to outvote a longer excursion, and matches
#: `calibration.N_SAMPLES_TEST_TIME` -- the perf gates converged on the same number against the
#: same kind of transient.
#:
#: **Widening the bar instead would have been the wrong fix.** The 8% exists to catch the two code
#: paths genuinely diverging; a bar loose enough to absorb single-draw noise (>12%) would stop doing
#: that. The noise is the thing to remove, and it is removable by sampling -- the same conclusion the
#: perf gates reached with `calibration.N_SAMPLES_TEST_TIME`.
_EQUIV_SAMPLES = 9


@pytest.mark.parametrize("n", [1024, 2048])
def test_event_matches_paired_median_idiom(dev, n):
    """bench_timing's event window agrees with the hand-rolled paired-median idiom within 8%.

    What this protects: the two ways of timing the same kernel must not drift apart. `bench_timing`
    is the repo's one trusted timer, and the paired-median idiom is what the design docs' numbers
    were originally produced with, so a divergence means one of them silently changed meaning.

    Each side is a median of `_EQUIV_SAMPLES` medians rather than a single one. That is a statement
    about the MEASUREMENT, not about the timer: at these sizes a single median varies ~12% run to
    run, so a single-draw comparison tests the GPU's mood rather than the two code paths. See
    `_EQUIV_SAMPLES` for the numbers.
    """
    import statistics

    fn = _gemm_fn(n)
    ref = statistics.median(
        _our_paired_median(fn, rounds=35, warmup=5, iters=20) for _ in range(_EQUIV_SAMPLES)
    )
    got = statistics.median(
        BU.benchmark_single(fn, rounds=35, warmup=5, iters=20, device=dev).median_ms
        for _ in range(_EQUIV_SAMPLES)
    )
    rel = abs(got - ref) / ref
    assert rel <= 0.08, (
        f"n={n}: event {got:.5f} vs paired-median idiom {ref:.5f} rel={rel:.2%} > 8% "
        f"(each a median of {_EQUIV_SAMPLES} runs, so this is a real divergence between the two "
        f"timing paths rather than single-draw noise)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# D. paired — drift-cancelled ratio matches the independent-single ratio; a
#    failing target is dropped without aborting the sweep.
# ─────────────────────────────────────────────────────────────────────────────
def test_paired_ratio_matches_single(dev):
    """benchmark_paired median_ratio ~= ratio of independent benchmark_single medians."""
    a = _gemm_fn(1024)
    b = _gemm_fn(768)
    ma = BU.benchmark_single(a, rounds=30, warmup=5, iters=20, device=dev).median_ms
    mb = BU.benchmark_single(b, rounds=30, warmup=5, iters=20, device=dev).median_ms
    single_ratio = ma / mb
    res = BU.benchmark_paired(
        {"a": a}, baseline=b, baseline_label="b", rounds=30, warmup=5, iters=20, device=dev
    )
    paired_ratio = res.targets["a"]["median_ratio"]
    assert paired_ratio is not None
    rel = abs(paired_ratio - single_ratio) / single_ratio
    assert rel <= 0.15, (
        f"paired ratio {paired_ratio:.3f} vs single ratio {single_ratio:.3f} rel={rel:.2%}"
    )


def test_paired_drops_failing_target(dev):
    """A target closure that raises is recorded status!='ok'; good targets still returned."""
    good = _gemm_fn(512)

    def bad():
        raise RuntimeError("intentional target failure")

    res = BU.benchmark_paired({"good": good, "bad": bad}, rounds=10, warmup=2, iters=5, device=dev)
    assert res.targets["good"]["status"] == "ok"
    assert res.targets["good"]["median_ms"] > 0
    assert res.targets["bad"]["status"] != "ok"


# ─────────────────────────────────────────────────────────────────────────────
# E. time_callable — the atomic single-window timer.
# ─────────────────────────────────────────────────────────────────────────────
def test_time_callable_positive(dev):
    fn = _gemm_fn(512)
    ms = BU.time_callable(fn, 10, dev)
    assert ms > 0 and math.isfinite(ms)


def test_time_callable_iters_scaling(dev):
    """ms/call (divided by iters) is stable across iters for a device-bound kernel.

    Validates the ``/ iters`` normalization: a device-bound GEMM's per-call time
    must not blow up or collapse as the back-to-back count grows, and the
    host-tax-heavy iters=1 sample stays at/above the amortized iters=50 sample.
    """
    fn = _gemm_fn(2048)
    t1 = BU.time_callable(fn, 1, dev)
    t10 = BU.time_callable(fn, 10, dev)
    t50 = BU.time_callable(fn, 50, dev)
    assert t1 > 0 and t10 > 0 and t50 > 0
    assert abs(t50 - t10) / t10 <= 0.25, (
        f"per-call unstable across iters: t10={t10:.5f} t50={t50:.5f}"
    )
    assert t1 >= t50 * 0.9, f"iters=1 {t1:.5f} should be >= amortized iters=50 {t50:.5f}"


# ─────────────────────────────────────────────────────────────────────────────
# F. error / guard paths — real pytest.raises (deterministic).
# ─────────────────────────────────────────────────────────────────────────────
def test_iters_lt_1_raises(dev):
    fn = _gemm_fn(256)
    with pytest.raises(ValueError):
        BU.time_callable(fn, 0, dev)


def test_time_callable_cpu_raises():
    """CUDA-event timing on a CPU device is a RuntimeError (needs no CUDA itself)."""
    x = torch.randn(4, 4)

    def fn():
        return x + x

    with pytest.raises(RuntimeError):
        BU.time_callable(fn, 1, torch.device("cpu"))


def test_invalid_mode_raises(dev):
    fn = _gemm_fn(256)
    with pytest.raises(ValueError):
        BU.benchmark_single(fn, rounds=2, warmup=1, iters=2, device=dev, mode="bogus")


def test_stream_not_restored_raises(dev):
    """A target that switches the current stream without restoring it must raise.

    CUDA events bound to one stream do not measure work on another, so
    time_callable asserts the current stream is unchanged after the window.
    """
    fn0 = _gemm_fn(256)
    default_stream = torch.cuda.current_stream(dev)
    other = torch.cuda.Stream(device=dev)

    def switcher():
        fn0()
        torch.cuda.set_stream(other)  # switch and DON'T restore

    try:
        with pytest.raises(RuntimeError):
            BU.time_callable(switcher, 1, dev)
    finally:
        # Restore so subsequent tests in this process are not corrupted.
        torch.cuda.set_stream(default_stream)


# ─────────────────────────────────────────────────────────────────────────────
# F2. reduce= param (event-mode consensus) — single-proc + error/guard paths.
#     The distributed consensus behavior is exercised in group G.
# ─────────────────────────────────────────────────────────────────────────────
def test_reduce_invalid_raises(dev):
    """An unknown reduce= value is a ValueError (validated up front, before timing)."""
    fn = _gemm_fn(256)
    with pytest.raises(ValueError):
        BU.benchmark_single(fn, rounds=2, warmup=1, iters=2, device=dev, reduce="banana")


def test_reduce_max_single_proc_noop(dev):
    """Single-proc reduce="max" is an identity no-op: median_ms == median(raw_ms).

    ``all_reduce_max`` is gated on ``is_distributed``; with world_size==1 the returned
    median must equal this process's own per-round median (the reduce changed nothing),
    exactly matching what reduce=None returns.
    """
    fn = _gemm_fn(512)
    r_max = BU.benchmark_single(
        fn, rounds=20, warmup=5, iters=10, device=dev, mode="event", reduce="max"
    )
    assert r_max.median_ms == statistics.median(r_max.raw_ms), (
        f"single-proc reduce=max altered the median: {r_max.median_ms} != "
        f"median(raw)={statistics.median(r_max.raw_ms)}"
    )
    r_none = BU.benchmark_single(
        fn, rounds=20, warmup=5, iters=10, device=dev, mode="event", reduce=None
    )
    assert r_none.median_ms == statistics.median(r_none.raw_ms)


def test_reduce_max_with_device_mode_allowed(dev):
    """reduce="max" + mode="device" is ALLOWED (device already MAX-reduces); returns finite.

    We chose to ACCEPT the redundant flag rather than raise: mode="device" always
    all_reduce(MAX)-reduces the per-rank median, so reduce="max" merely restates the
    built-in behavior. Only genuinely-invalid reduce= values raise.
    """
    fn = _gemm_fn(256)
    res = BU.benchmark_single(
        fn, rounds=10, warmup=3, iters=5, device=dev, mode="device", reduce="max"
    )
    assert math.isfinite(res.median_ms) and res.median_ms > 0


# ─────────────────────────────────────────────────────────────────────────────
# H. resolve_dist — the three resolution sources + the fields they populate.
# ─────────────────────────────────────────────────────────────────────────────
class _FakeManager:
    """Duck-typed manager exposing .device/.rank/.world_size/.local_rank/.barrier()."""

    def __init__(self, device, rank, world_size, local_rank=0):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.barrier_calls = 0

    def barrier(self):
        self.barrier_calls += 1


class _MinimalManager:
    """A manager exposing ONLY .device/.rank (world_size/local_rank fall back to defaults)."""

    def __init__(self, device, rank):
        self.device = device
        self.rank = rank


def test_resolve_dist_adapter_passthrough(dev):
    """An already-built _DistAdapter is returned AS-IS (identity, not rewrapped)."""
    a = BU.resolve_dist(None, device=dev)
    assert BU.resolve_dist(a) is a


def test_resolve_dist_from_manager_populates_fields(dev):
    mgr = _FakeManager(dev, rank=2, world_size=4, local_rank=1)
    da = BU.resolve_dist(mgr)
    assert da.device == dev and da.rank == 2 and da.world_size == 4
    assert da.is_distributed is True
    assert da.is_rank0 is False
    da.barrier()
    assert mgr.barrier_calls == 1  # the manager's OWN barrier() is preferred over torch.distributed


def test_resolve_dist_manager_missing_fields_defaults(dev):
    """A manager exposing only .device/.rank: world_size defaults to 1 (single-process, not distributed)."""
    mgr = _MinimalManager(dev, rank=0)
    da = BU.resolve_dist(mgr)
    assert da.world_size == 1
    assert da.is_distributed is False
    assert da.is_rank0 is True


def test_resolve_dist_none_uses_explicit_device_override():
    """dist=None with an explicit device= override (no live torch.distributed group in this process)."""
    assert not torch.distributed.is_initialized()
    cpu = torch.device("cpu")
    da = BU.resolve_dist(None, device=cpu)
    assert da.device == cpu and da.rank == 0 and da.world_size == 1
    assert da.is_distributed is False and da.is_rank0 is True


def test_resolve_dist_none_defaults_to_current_cuda_device(dev):
    da = BU.resolve_dist(None)
    assert da.device == dev
    assert da.world_size == 1 and da.rank == 0 and not da.is_distributed


# ─────────────────────────────────────────────────────────────────────────────
# I. benchmark_paired — baseline field population, no-baseline shape, ordering, and a raising baseline.
# ─────────────────────────────────────────────────────────────────────────────
def test_paired_baseline_median_ms_field_matches_independent_single(dev):
    a = _gemm_fn(512)
    b = _gemm_fn(384)
    res = BU.benchmark_paired(
        {"a": a}, baseline=b, baseline_label="b", rounds=20, warmup=4, iters=10, device=dev
    )
    assert res.baseline_label == "b"
    assert res.baseline_median_ms is not None and res.baseline_median_ms > 0
    single_b = BU.benchmark_single(b, rounds=20, warmup=4, iters=10, device=dev).median_ms
    rel = abs(res.baseline_median_ms - single_b) / single_b
    assert rel <= 0.25, (
        f"paired baseline_median_ms {res.baseline_median_ms:.5f} vs single {single_b:.5f}"
    )


def test_paired_no_baseline_ratio_and_baseline_median_are_none(dev):
    a = _gemm_fn(256)
    res = BU.benchmark_paired({"a": a}, rounds=8, warmup=2, iters=5, device=dev)
    assert res.baseline_label is None
    assert res.baseline_median_ms is None
    assert res.targets["a"]["median_ratio"] is None
    assert res.targets["a"]["status"] == "ok"
    assert res.targets["a"]["median_ms"] > 0


def test_paired_baseline_raise_propagates_uncaught(dev):
    """A baseline closure that raises is NOT caught by the per-target drop mechanism (only targets are
    isolated) — the exception propagates out of benchmark_paired. Documents the real (asymmetric) contract:
    a bad baseline aborts the whole sweep, a bad TARGET does not."""
    good = _gemm_fn(256)

    def bad_baseline():
        raise RuntimeError("baseline exploded")

    with pytest.raises(RuntimeError, match="baseline exploded"):
        BU.benchmark_paired(
            {"good": good}, baseline=bad_baseline, rounds=5, warmup=1, iters=5, device=dev
        )


def test_paired_print_report_orders_targets_by_median_ascending(dev, capsys):
    """_print_paired lists targets fastest-first; a multi-target report must reflect the real ordering."""
    slow = _gemm_fn(1536)
    fast = _gemm_fn(128)
    BU.benchmark_paired(
        {"slow": slow, "fast": fast}, rounds=15, warmup=3, iters=5, device=dev, print_report=True
    )
    out = capsys.readouterr().out
    assert "fast" in out and "slow" in out
    assert out.index("fast") < out.index("slow")


# ─────────────────────────────────────────────────────────────────────────────
# J. benchmark_single(mode="device") — the iters -> ms-budget mapping (warmup_ms/rep_ms + floors) and
#    min/std/raw population, isolated from real timing via a stubbed _do_bench_native.
# ─────────────────────────────────────────────────────────────────────────────
def test_device_mode_budget_mapping_and_result_population(dev, monkeypatch):
    captured = {}

    def _fake_do_bench_native(fn, *, warmup_ms, rep_ms, dev, da):
        captured["warmup_ms"] = warmup_ms
        captured["rep_ms"] = rep_ms
        return [1.0, 1.1, 0.9]

    monkeypatch.setattr(BU, "_do_bench_native", _fake_do_bench_native)
    fn = _gemm_fn(64)
    res = BU.benchmark_single(fn, rounds=10, warmup=3, iters=20, device=dev, mode="device")
    assert captured["warmup_ms"] == max(3 * 5.0, 25.0)
    assert captured["rep_ms"] == max(10 * 20 * 0.05, 20.0)
    assert res.raw_ms == [1.0, 1.1, 0.9]
    assert res.min_ms == 0.9
    assert res.std_ms == pytest.approx(statistics.pstdev([1.0, 1.1, 0.9]))
    assert res.rounds == 3 and res.iters == 1
    assert res.median_ms == pytest.approx(statistics.median([1.0, 1.1, 0.9]))


def test_device_mode_budget_floors_apply_for_tiny_args(dev, monkeypatch):
    captured = {}

    def _fake_do_bench_native(fn, *, warmup_ms, rep_ms, dev, da):
        captured["warmup_ms"] = warmup_ms
        captured["rep_ms"] = rep_ms
        return [2.0]

    monkeypatch.setattr(BU, "_do_bench_native", _fake_do_bench_native)
    fn = _gemm_fn(64)
    BU.benchmark_single(fn, rounds=1, warmup=1, iters=1, device=dev, mode="device")
    assert captured["warmup_ms"] == 25.0  # floor: max(1*5, 25)
    assert captured["rep_ms"] == 20.0  # floor: max(1*1*0.05, 20)


def test_device_mode_adaptive_rep_count_contract_documented():
    """Doc-only guard (can't provoke a real collective deadlock in a unit test): the docstring MUST state
    the adaptive-rep-count / collective-unsafety contract and point at the safe alternative."""
    doc = BU.benchmark_single.__doc__
    assert "ADAPTIVE" in doc
    assert 'mode="event", reduce="max"' in doc


def test_event_mode_raw_ms_length_is_lockstep_with_rounds(dev):
    """event mode's raw_ms always has EXACTLY `rounds` samples (fixed-N, unlike device mode's adaptive
    rep-count) — the property a distributed autotuner of in-kernel-collective kernels relies on."""
    fn = _gemm_fn(256)
    res = BU.benchmark_single(fn, rounds=9, warmup=2, iters=4, device=dev, mode="event")
    assert len(res.raw_ms) == 9 == res.rounds


# ─────────────────────────────────────────────────────────────────────────────
# K. warmup-drift guard — direct, deterministic unit tests of _maybe_warn_warmup_drift.
# ─────────────────────────────────────────────────────────────────────────────
def test_warmup_drift_warns_when_warmup_much_slower(dev, capsys):
    da = BU.resolve_dist(None, device=dev)
    BU._maybe_warn_warmup_drift([10.0, 10.0, 10.0], [1.0, 1.0, 1.0, 1.0], da)
    assert "warmup may be insufficient" in capsys.readouterr().err


def test_warmup_drift_warns_when_warmup_much_faster(dev, capsys):
    """Two-sided: a suspiciously FAST warmup vs bench also fires (cache-cold bench tell)."""
    da = BU.resolve_dist(None, device=dev)
    BU._maybe_warn_warmup_drift([0.1, 0.1, 0.1], [1.0, 1.0, 1.0, 1.0], da)
    assert "warmup may be insufficient" in capsys.readouterr().err


def test_warmup_drift_silent_when_stable(dev, capsys):
    da = BU.resolve_dist(None, device=dev)
    BU._maybe_warn_warmup_drift([1.0, 1.0, 1.0], [1.05, 0.98, 1.02, 1.0], da)
    assert capsys.readouterr().err == ""


def test_warmup_drift_skips_below_minimum_sample_counts(capsys):
    """< 1 warmup sample or < 3 bench samples -> early return, no warning (even with huge drift)."""
    da = BU.resolve_dist(None, device=torch.device("cpu"))
    BU._maybe_warn_warmup_drift([], [1.0, 1.0, 1.0], da)  # 0 warmup samples
    assert capsys.readouterr().err == ""
    BU._maybe_warn_warmup_drift(
        [10.0], [1.0, 1.0], da
    )  # bench_ms has < 3 samples, huge drift otherwise
    assert capsys.readouterr().err == ""


# ─────────────────────────────────────────────────────────────────────────────
# L. time_callable — /iters exactness, the stream= param, and the sync-before/after discipline.
# ─────────────────────────────────────────────────────────────────────────────
def test_time_callable_iters_normalization_is_exact(dev, monkeypatch):
    """Stub CUDA-event timing to a FIXED window so `/ iters` is checked as exact arithmetic, not jitter."""
    monkeypatch.setattr(torch.cuda.Event, "record", lambda self, stream=None: None)
    monkeypatch.setattr(torch.cuda.Event, "elapsed_time", lambda self, other: 123.0)
    ms = BU.time_callable(lambda: None, 7, dev)
    assert ms == pytest.approx(123.0 / 7)
    ms2 = BU.time_callable(lambda: None, 41, dev)
    assert ms2 == pytest.approx(123.0 / 41)


def test_time_callable_explicit_stream_param_used_when_ambient_matches(dev):
    """Passing stream= succeeds when the caller has ALSO made it the ambient current stream (the intended
    usage — the caller wraps the launch in `with torch.cuda.stream(s):`)."""
    fn = _gemm_fn(128)
    custom = torch.cuda.Stream(device=dev)
    with torch.cuda.stream(custom):
        ms = BU.time_callable(fn, 5, dev, stream=custom)
    assert ms > 0 and math.isfinite(ms)


def test_time_callable_explicit_stream_param_mismatch_raises(dev):
    """Passing stream= WITHOUT making it ambient is exactly the "switched stream, didn't restore" case
    time_callable guards against: the post-loop current-stream check fails and it raises."""
    custom = torch.cuda.Stream(device=dev)
    with pytest.raises(RuntimeError):
        BU.time_callable(lambda: None, 3, dev, stream=custom)


def test_time_callable_syncs_before_start_and_after_stop(dev, monkeypatch):
    """Discipline (c): a synchronize() brackets the window on BOTH sides (before start.record, after
    end.record) so a stale enqueue can't bleed in and the stop event is drained before elapsed_time()."""
    calls = []
    orig_sync = torch.cuda.synchronize

    def _spy(device=None):
        calls.append(device)
        return orig_sync(device)

    monkeypatch.setattr(torch.cuda, "synchronize", _spy)
    BU.time_callable(lambda: None, 3, dev)
    assert len(calls) >= 2, f"expected >=2 synchronize() calls (before+after), got {len(calls)}"


# ─────────────────────────────────────────────────────────────────────────────
# G. distributed (torchrun-collected; SKIPs when world_size==1 via dist_env).
# ─────────────────────────────────────────────────────────────────────────────
def test_distributed_event_mode_runs(dist_env):
    """Event mode under torchrun returns a finite positive median (barrier path, no deadlock)."""
    fn = _gemm_fn(256, device=dist_env.device)
    res = BU.benchmark_single(fn, rounds=10, warmup=3, iters=5, device=dist_env.device)
    assert math.isfinite(res.median_ms) and res.median_ms > 0


def test_distributed_device_mode_runs(dist_env):
    """The NEW native device path runs distributed (was single-GPU-only pre-Phase-2)."""
    fn = _gemm_fn(256, device=dist_env.device)
    res = BU.benchmark_single(
        fn, rounds=10, warmup=3, iters=5, device=dist_env.device, mode="device"
    )
    assert math.isfinite(res.median_ms) and res.median_ms > 0


def test_distributed_consensus_all_reduce_max(dist_env):
    """The reduced median_ms is IDENTICAL on every rank and equals the slowest (MAX).

    A per-rank size skew makes higher ranks strictly slower, so the all_reduce(MAX)
    consensus must land on the top rank's median on ALL ranks.
    """
    import torch.distributed as dist

    # Skew: rank r runs a larger GEMM, so rank world_size-1 is the slowest.
    n = 256 + 128 * dist_env.rank
    fn = _gemm_fn(n, device=dist_env.device)
    res = BU.benchmark_single(
        fn, rounds=15, warmup=3, iters=5, device=dist_env.device, mode="device"
    )
    gathered = [None] * dist_env.world_size
    dist.all_gather_object(gathered, res.median_ms)
    assert len(set(gathered)) == 1, f"median_ms not consensus across ranks: {gathered}"
    # MAX(consensus) == this rank's returned value (all equal to the slowest rank's).
    assert res.median_ms == max(gathered), f"consensus {res.median_ms} != max {max(gathered)}"


def test_distributed_event_reduce_max_consensus(dist_env):
    """event mode + reduce="max": returned median_ms is bit-identical AND == MAX(local medians).

    The fixed-N barrier'd counterpart to the device-mode consensus test: a per-rank size
    skew makes the top rank strictly slowest, so the all_reduce(MAX) must land on the
    slowest rank's LOCAL median on ALL ranks. Unlike the device path this stays lockstep
    for a fixed iters (the property the distributed autotuner relies on). We verify against
    the SAME run's per-rank local medians (median(raw_ms)), so no cross-run jitter.
    """
    import torch.distributed as dist

    # Skew: rank r runs a larger GEMM, so rank world_size-1 is the slowest.
    n = 256 + 128 * dist_env.rank
    fn = _gemm_fn(n, device=dist_env.device)
    res = BU.benchmark_single(
        fn, rounds=15, warmup=3, iters=5, device=dist_env.device, mode="event", reduce="max"
    )
    # In event mode raw_ms stays per-rank, so its median IS the local median that got reduced.
    local_med = statistics.median(res.raw_ms)
    local_meds = [None] * dist_env.world_size
    dist.all_gather_object(local_meds, local_med)
    reduced = [None] * dist_env.world_size
    dist.all_gather_object(reduced, res.median_ms)
    # (1) consensus: every rank's returned median is bit-identical.
    assert len(set(reduced)) == 1, f"reduce=max median not consensus across ranks: {reduced}"
    # (2) it equals MAX over the SAME run's per-rank local medians (bit-exact: float64 MAX
    #     of already-float64 medians round-trips exactly).
    assert res.median_ms == max(local_meds), (
        f"reduce=max median {res.median_ms} != MAX(local medians) {max(local_meds)}"
    )
    # (3) the reduce is non-trivial: this rank's own local median never exceeds the consensus
    #     (the slowest rank's local median equals it → MAX truly picked the straggler).
    assert local_med <= res.median_ms, f"local {local_med} > consensus {res.median_ms}"


def test_distributed_event_reduce_none_stays_per_rank(dist_env):
    """reduce=None (default) leaves the median PER-RANK: with a size skew ranks disagree.

    The negative control for reduce="max": proves the reduce is what produces consensus,
    not the event path itself. Needs >=2 ranks and a real per-rank skew so the medians
    are measurably distinct.
    """
    if dist_env.world_size < 2:
        pytest.skip("per-rank divergence needs >= 2 ranks")
    import torch.distributed as dist

    # Strong skew (n doubles per rank) so the un-reduced medians are clearly distinct.
    n = 512 * (dist_env.rank + 1)
    fn = _gemm_fn(n, device=dist_env.device)
    res = BU.benchmark_single(
        fn, rounds=20, warmup=5, iters=20, device=dist_env.device, mode="event", reduce=None
    )
    gathered = [None] * dist_env.world_size
    dist.all_gather_object(gathered, res.median_ms)
    assert len(set(gathered)) > 1, (
        f"reduce=None should stay per-rank but all ranks agree: {gathered}"
    )


# ── compare_cold_compile: the sampling discipline for a non-device measurement ─────────────────
# These need no GPU and no subprocess: `compare_cold_compile` consumes a `measure` callable, so a
# test drives it with literal numbers. That is the point of the design -- see its docstring.


def _canned(**per_side):
    """A ``measure`` callable that hands back a prepared sequence per side.

    Args:
        **per_side: label -> the durations that side's calls return, IN CALL ORDER. Because
            `compare_cold_compile` alternates ABBA, the nth value of a side is consumed on that
            side's nth call, which is what the tests below assert against.

    Returns:
        ``measure(label) -> float``, raising ``AssertionError`` if a side is called more times than
        it has values -- a silent wrap-around would make an alternation bug look like a pass.
    """
    it = {k: iter(v) for k, v in per_side.items()}

    def measure(label):
        try:
            return next(it[label])
        except StopIteration:  # pragma: no cover - only on an alternation bug
            raise AssertionError(f"{label!r} was measured more times than it has canned values")

    return measure


@pytest.mark.parametrize("samples", [1, 2, 3, 4])
def test_cold_compile_refuses_fewer_than_five_samples(samples):
    """Below the floor it RAISES, rather than measuring something that cannot be trusted.

    Refusing is the whole mechanism. A permissive version that merely warned would be satisfied by
    the exact one-sample comparison that produced a false blocker twice, because nobody reads a
    warning attached to a number that confirms what they feared.
    """
    with pytest.raises(ValueError, match=r"below the floor of 5 per side"):
        BU.compare_cold_compile(_canned(main=[1.0] * 9, ours=[1.0] * 9), samples=samples)


def test_cold_compile_accepts_the_floor_exactly():
    """Five is admissible -- the floor is a minimum, not a value to exceed."""
    res = BU.compare_cold_compile(_canned(main=[1.0] * 5, ours=[2.0] * 5), samples=5)
    assert res.samples == 5
    assert res.baseline_median_s == 1.0 and res.candidate_median_s == 2.0


def test_cold_compile_alternates_the_two_sides_in_abba_order():
    """The realized call order is ABBA, and it is reported so a reader can verify it.

    Plain ABAB would still give one side the first-after-a-gap slot in every pair, and that slot
    pays page-cache and import costs the second does not. ABBA rotates it, so a positional bias
    lands on both sides equally instead of on one.
    """
    res = BU.compare_cold_compile(_canned(main=[1.0] * 6, ours=[1.0] * 6), samples=6)
    # One round per pair, six rounds, the leading side swapping every round.
    assert res.order == (
        # fmt: off
        "main",
        "ours",
        "ours",
        "main",
        "main",
        "ours",
        "ours",
        "main",
        "main",
        "ours",
        "ours",
        "main",
        # fmt: on
    )
    assert len(res.order) == 2 * res.samples


def test_cold_compile_takes_the_median_and_not_the_mean():
    """One long tail sample moves a mean of five by a fifth of its excess and the median not at all.

    This is the failure mode a cold compile actually has: a page-cache miss or another job landing
    on the box inflates ONE process. Reporting a mean would carry that straight into the delta.
    """
    res = BU.compare_cold_compile(
        _canned(main=[6.0, 6.0, 6.0, 6.0, 6.0], ours=[6.0, 6.0, 6.0, 6.0, 30.0]), samples=5
    )
    assert res.candidate_median_s == 6.0, "the median must ignore the tail"
    assert statistics.mean(res.candidate_samples) == pytest.approx(10.8), (
        "the mean is what a median is being preferred over; if this stops being 10.8 the fixture "
        "no longer exercises the distinction"
    )
    assert res.delta_s == 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_cold_compile_refuses_a_duration_that_is_not_a_compile(bad):
    """Zero, negative, infinite and NaN are refused at the sample, not laundered into a median.

    A zero-second "cold compile" means the measurement did not measure one -- a warm cache, a
    mis-parsed log line, or a process that died before compiling. Folded into a median of five it
    would produce a perfectly plausible number with nothing to indicate it was fiction.
    """
    with pytest.raises(ValueError, match=r"did not measure a compile"):
        BU.compare_cold_compile(_canned(main=[1.0] * 5, ours=[1.0, bad, 1.0, 1.0, 1.0]), samples=5)


def test_cold_compile_refuses_two_sides_with_one_name():
    """Two sides sharing a label would make the returned samples unattributable."""
    with pytest.raises(ValueError, match=r"both 'x'"):
        BU.compare_cold_compile(_canned(x=[1.0] * 10), baseline="x", candidate="x", samples=5)


# The numbers below are RECONSTRUCTED, not raw. The source report gave, for `N=256 D=128 cont_pipe`: a single
# sample reading ours 9.037 s against main 6.074 s (+2.96 s, +49%, "we regressed"), and five
# samples per side giving medians 6.364 and 6.292 (+0.072 s, +1.1%, parity). The individual ten
# durations were not reported, so each side below carries its REPORTED first sample and its
# REPORTED median, with the remaining three chosen only to be consistent with both. Every assertion
# here is against a number the source actually published; nothing depends on the filler.
_W1_MAIN = [6.074, 6.250, 6.292, 6.310, 6.350]
_W1_OURS = [9.037, 6.300, 6.350, 6.364, 6.400]


def test_cold_compile_turns_w1s_false_regression_into_the_parity_answer():
    """The measurement this primitive exists for: +49% at one sample, +1.1% at five.

    Under this repo's bring-back rule a cold-compile regression is a BLOCKER, so the one-sample
    answer does not merely mislead -- it stops a good commit and sends someone hunting a bug that
    is not there. Both readings are asserted here, because the point is not that five samples give
    a smaller number; it is that the two disagree by a factor that changes the decision.
    """
    one_sample_delta = _W1_OURS[0] - _W1_MAIN[0]
    assert one_sample_delta == pytest.approx(2.963, abs=1e-3)
    assert one_sample_delta / _W1_MAIN[0] == pytest.approx(0.488, abs=1e-3), (
        "+49%, the false blocker"
    )

    res = BU.compare_cold_compile(_canned(main=_W1_MAIN, ours=_W1_OURS), samples=5)
    assert res.baseline_median_s == pytest.approx(6.292)
    assert res.candidate_median_s == pytest.approx(6.364)
    assert res.delta_s == pytest.approx(0.072, abs=1e-3)
    assert res.ratio == pytest.approx(1.0114, abs=1e-3), "+1.1%, parity"


def test_cold_compile_calls_w1s_delta_unresolvable():
    """+0.072 s against a 2.7 s observed range is not a delta anyone should act on.

    ``resolvable`` is the fact a single sample cannot supply, and it is the one that would have
    stopped the false blocker at the moment it was measured rather than after the second opinion.
    False here does NOT claim the two sides are equal -- it says this run cannot tell them apart.
    """
    res = BU.compare_cold_compile(_canned(main=_W1_MAIN, ours=_W1_OURS), samples=5)
    assert res.candidate_spread_s == pytest.approx(2.737, abs=1e-3)
    assert abs(res.delta_s) < res.candidate_spread_s
    assert res.resolvable is False


def test_cold_compile_resolves_a_shift_that_is_larger_than_the_noise():
    """The other direction: a real, tight regression IS reported as resolvable.

    Without this the ``resolvable`` flag would be satisfied by always returning False, which is the
    degenerate way to never raise a false blocker and also never catch a true one.
    """
    res = BU.compare_cold_compile(
        _canned(main=[6.00, 6.02, 6.01, 6.03, 6.01], ours=[9.00, 9.02, 9.01, 9.03, 9.01]),
        samples=5,
    )
    assert res.delta_s == pytest.approx(3.0, abs=0.05)
    assert res.resolvable is True


def test_cold_compile_reports_both_sides_and_the_spread_not_just_a_ratio():
    """``describe()`` carries both medians, both spreads and the realized sample lists.

    A single ratio hides which side moved and how noisy either was, which is precisely what made
    the one-sample number look authoritative. The report is meant to be pasted verbatim, so its
    content is part of the contract.
    """
    res = BU.compare_cold_compile(_canned(main=_W1_MAIN, ours=_W1_OURS), samples=5)
    text = res.describe()
    assert "main" in text and "ours" in text
    assert "6.292" in text and "6.364" in text, "both medians"
    assert "spread" in text and "2.737" in text, "the spread, which is the whole lesson"
    assert "resolvable=False" in text
    assert "5 samples/side, alternated" in text
    assert "9.037" in text, "the raw samples, so a reader can see the outlier for themselves"


def test_the_mad_survives_an_outlier_the_range_does_not():
    """The case the MAD was added for: one slow compile makes the RANGE useless, not the side noisy.

    `resolvable` is gated on ``max - min``, so a single excursion out of five inflates it and a real
    delta reports as "cannot tell at this sample count" -- the fail-dangerous direction on a bar
    that has never had a number. The MAD is reported beside it so a reader can see that the two
    disagree, which is the signal that a False `resolvable` means "one outlier", not "parity".

    Neither replaces the other, and that is measured rather than argued: the tight-with-outlier
    series below has a range 20x its MAD, while a genuinely dispersed series has the two within a
    factor of ~2. A rule on either scalar alone cannot tell those apart.
    """
    from fold_cp_ops._internal.bench_timing import ColdCompileComparison

    tight_with_outlier = (10.0, 10.1, 10.0, 10.1, 12.0)  # four tight, one excursion
    genuinely_noisy = (10.0, 11.0, 12.0, 13.0, 14.0)  # dispersed throughout
    cmp_ = ColdCompileComparison(
        baseline="main",
        candidate="ours",
        baseline_samples=tight_with_outlier,
        candidate_samples=genuinely_noisy,
        order=("main", "ours", "ours", "main") * 2,
    )
    assert cmp_.baseline_spread_s == pytest.approx(2.0), "the range sees the whole excursion"
    assert cmp_.baseline_mad_s == pytest.approx(0.1), "the MAD does not"
    assert cmp_.baseline_spread_s / cmp_.baseline_mad_s > 10, (
        "they disagree by an order of magnitude"
    )

    # and on a genuinely dispersed side they AGREE, which is what makes the disagreement above
    # informative rather than a property of the statistic
    assert cmp_.candidate_spread_s == pytest.approx(4.0)
    assert cmp_.candidate_mad_s == pytest.approx(1.0)
    assert cmp_.candidate_spread_s / cmp_.candidate_mad_s < 5


def test_the_verdict_is_unchanged_by_the_new_fields():
    """`resolvable` still means exactly what it meant -- this change is ADDITIVE.

    Adding a primitive must not silently redefine an existing verdict. `resolvable` is consumed by
    `describe`, `__repr__`, the cold-compile payload and three tests. Pinning its formula here stops
    a later change from altering a life-line's meaning unintentionally.
    """
    from fold_cp_ops._internal.bench_timing import ColdCompileComparison

    c = ColdCompileComparison(
        baseline="main",
        candidate="ours",
        baseline_samples=(10.0, 10.1, 10.0, 10.1, 12.0),
        candidate_samples=(10.0, 10.1, 10.0, 10.1, 10.0),
        order=("main", "ours", "ours", "main") * 2,
    )
    # delta is small; the RANGE (2.0) dwarfs it, so the verdict is False -- the very case the MAD
    # (0.1) shows to be one outlier rather than parity.
    assert abs(c.delta_s) < c.baseline_spread_s
    assert c.resolvable is False
    assert c.resolvable == (abs(c.delta_s) > max(c.baseline_spread_s, c.candidate_spread_s))
    assert "mad" in c.describe() and "gated on the RANGE" in c.describe()


def test_a_single_sample_reports_zero_dispersion_and_that_is_not_stability():
    """One point exhibits no dispersion. The number is honest; reading it as "stable" is not.

    Pinned because a zero here looks exactly like a very tight side, and `compare_cold_compile`'s
    five-sample floor is the thing that stops a caller ever seeing it -- a floor is easier to
    relax than to re-derive, so the degenerate value is documented rather than left to be found.
    """
    from fold_cp_ops._internal.bench_timing import ColdCompileComparison

    assert ColdCompileComparison._mad_s((7.0,)) == 0.0
    assert ColdCompileComparison._mad_s(()) == 0.0


# ── the per-rank timing diagnostic (RANK_DIAG_ENV) ────────────────────────────────────────────
#
# WHY THIS EXISTS. `all_reduce(MAX)` takes the per-rank value as an ARGUMENT and returns only the
# maximum, so nothing downstream can tell "every rank was slow" from "one rank was slow and MAX
# reported it" -- opposite verdicts for a perf pin. Measured consequence: a pin file whose schema is
# `['key','median_ms','rel_std']` cannot express the difference, and `rel_std` does not recover it
# either, because it is the spread of the ALREADY-REDUCED samples -- a consistently slow rank yields
# a TIGHT rel_std around an INFLATED median, which is what a clean cell also looks like.


def _patch_gather(monkeypatch, per_rank):
    """Make the REAL `all_gather_max_diag` run on this single process, over a canned gather.

    Purpose
        Exercise the method's own argmax computation. The obvious alternative -- a subclass that
        overrides `all_gather_max_diag` and returns a canned tuple -- was WRITTEN FIRST AND WAS
        VACUOUS: hardcoding the real method's argmax to 0 left every test green, because the fake
        had reimplemented the exact line under test. Patching the COLLECTIVE instead leaves the
        method body intact, so the mutation is visible.

    Semantics
        Patches `torch.distributed.is_available` / `is_initialized` to True and replaces
        `all_gather_into_tensor` with a fill from ``per_rank``. Nothing is sent anywhere; there is no
        process group. Undone by monkeypatch at teardown.

    Args:
        monkeypatch: pytest's fixture.
        per_rank: One float per rank, index == rank. ``len`` must equal the adapter's world size
            the caller builds, or the fill writes the wrong number of elements and the test is
            asserting against a shape it did not intend.

    Returns:
        None.
    """

    def _fill(buf, src, *a, **k):
        buf.copy_(torch.tensor(per_rank, dtype=buf.dtype, device=buf.device))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", _fill)


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
def test_rank_diag_enabled_accepts_any_truthy_spelling(monkeypatch, value):
    """Any non-empty, non-``0``/``false``/``no`` value enables it.

    Deliberately permissive: a diagnostic that stayed silently OFF because the operator wrote
    ``true`` instead of ``1`` is worse than one that is too eager, since the failure mode is a run
    that looks measured and is not.
    """
    monkeypatch.setenv(BU.RANK_DIAG_ENV, value)
    assert BU.rank_diag_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", " 0 "])
def test_rank_diag_disabled_by_default_and_by_falsey_spellings(monkeypatch, value):
    """Unset, empty, and the explicit falsey spellings all leave it OFF."""
    monkeypatch.delenv(BU.RANK_DIAG_ENV, raising=False)
    assert BU.rank_diag_enabled() is False
    monkeypatch.setenv(BU.RANK_DIAG_ENV, value)
    assert BU.rank_diag_enabled() is False


def test_consensus_max_off_path_never_gathers_and_reports_nothing(monkeypatch):
    """OFF: the shipped `all_reduce_max` runs, and the diagnostic fields are None, not rank 0.

    The None matters as much as the reduction. A placeholder ``argmax_rank=0`` would be
    indistinguishable from a measured rank 0 -- the precise confusion this diagnostic exists to
    remove -- so "not measured" has to be its own value.

    Non-vacuity is asserted rather than assumed: the gather is replaced with a raiser, so if the off
    path ever reached it this fails instead of quietly passing. Verified by mutation: removing the
    env gate fails this test.
    """
    monkeypatch.delenv(BU.RANK_DIAG_ENV, raising=False)
    da = BU._DistAdapter(torch.device("cpu"), 0, 3)
    monkeypatch.setattr(da, "all_reduce_max", lambda v: 99.0)  # only the SHIPPED path yields this
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        lambda *a, **k: pytest.fail("off path reached the gather: the env gate is not gating"),
    )
    value, argmax, per_rank = BU._consensus_max(da, 1.0)
    assert value == 99.0, "the off path did not go through all_reduce_max"
    assert argmax is None and per_rank is None, (
        f"the OFF path reported diagnostic fields (argmax={argmax}, per_rank={per_rank}); None is "
        "the only value that cannot be mistaken for a measured rank 0"
    )


def test_consensus_max_on_path_names_a_slowest_rank_that_is_not_zero(monkeypatch):
    """ON: the argmax names the rank that actually supplied the max, and the max is unchanged.

    **The slowest rank is 2, deliberately.** An implementation that hardcoded 0, or returned
    ``self.rank``, passes a test whose slowest rank happens to be 0 -- so the fixture puts it
    elsewhere and the assertion fails for both bugs. Verified by mutation: hardcoding the argmax to
    0 fails this test.

    **The collective is patched, not the method** (see `_patch_gather`), so the real
    `all_gather_max_diag` body runs. An earlier version subclassed and overrode that method, and the
    hardcoded-argmax mutation left it GREEN -- the fake had reimplemented the line under test.

    Also asserts the reported value is EXACTLY ``max(per_rank)``. MAX selects an input rather than
    accumulating, so the diagnostic must be bit-identical to the reduction it replaces; a diagnostic
    that changed the number would make its own runs unusable.
    """
    monkeypatch.setenv(BU.RANK_DIAG_ENV, "1")
    per = [1.25, 2.5, 7.75, 3.0]
    _patch_gather(monkeypatch, per)
    da = BU._DistAdapter(torch.device("cpu"), 1, len(per))
    value, argmax, got = BU._consensus_max(da, per[1])
    assert argmax == 2, f"argmax should name the rank holding {max(per)}, got rank {argmax}"
    assert value == max(per) == 7.75, f"the reduced value changed: {value} != {max(per)}"
    assert got == per, f"per-rank list is not the gathered one: {got} != {per}"


def test_consensus_max_is_identity_on_a_single_process(monkeypatch):
    """world_size==1 short-circuits in BOTH modes: no collective, and nothing claimed.

    Single-process is the standalone path every local bench takes, so the diagnostic must not turn
    it into a collective. `argmax_rank` stays None even with the env set: there is no other rank for
    it to have been, so reporting 0 would assert a measurement that never happened.
    """
    da = BU._DistAdapter(torch.device("cpu"), 0, 1)
    for setting in (None, "1"):
        if setting is None:
            monkeypatch.delenv(BU.RANK_DIAG_ENV, raising=False)
        else:
            monkeypatch.setenv(BU.RANK_DIAG_ENV, setting)
        value, argmax, per_rank = BU._consensus_max(da, 3.5)
        assert (value, argmax, per_rank) == (3.5, None, None), (
            f"single-process {BU.RANK_DIAG_ENV}={setting!r} did not short-circuit: "
            f"{(value, argmax, per_rank)}"
        )


def test_all_gather_max_diag_refuses_to_guess_without_a_group():
    """A distributed adapter with no live group RAISES rather than reporting its own rank.

    The tempting fallback -- return ``(value, self.rank, [value])`` -- produces a field that looks
    measured and is a guess, which is worse than the gap it fills. It raises instead, naming the env
    var to unset.
    """
    da = BU._DistAdapter(torch.device("cpu"), 0, 4)  # claims 4 ranks, no group initialized
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        pytest.skip("a live process group is present; this asserts the NO-group refusal")
    with pytest.raises(RuntimeError, match="live torch.distributed group"):
        da.all_gather_max_diag(1.0)


def test_bench_result_diagnostic_fields_default_to_none(dev):
    """A real single-process benchmark leaves both fields None, so nothing downstream changes.

    This is the compatibility half: every existing consumer builds and reads `BenchResult` without
    knowing about these fields, so they must default to None and stay None on the shipped path.
    """
    r = BU.benchmark_single(
        _gemm_fn(256), rounds=3, warmup=1, iters=2, device=dev, mode="event", reduce="max"
    )
    assert r.argmax_rank is None and r.per_rank_ms is None, (
        f"the shipped path populated diagnostic fields (argmax={r.argmax_rank}, "
        f"per_rank={r.per_rank_ms}); they must be None unless {BU.RANK_DIAG_ENV} is set"
    )


# NO @matrix_exempt on the two tests below, and that is a decision rather than an omission: the
# matrix audit is scoped by path PARTS to {kernels, perf, workflows, distributed} and
# `tests/_internal/` carries none of them, so this file is out of scope. Measured: zero occurrences
# of the decorator in this file, and `matrix_exempt` is not even imported -- adding one would be a
# NameError at import, which is how I nearly broke this file.
class _DiagStub:
    """Minimal duck-typed stand-in for `_DistAdapter`, so the emit can be tested with no group.

    Purpose
        `_consensus_max` needs only four things from its adapter, and building a real one requires a
        live process group -- which would make this test a distributed test and put it out of reach
        of the single-device suite where `bench_timing` is otherwise covered.

    Input requirements
        Nothing to supply: the returned triple is fixed so the assertions can name exact values. A
        caller who changes `per_rank` must change the expected `argmax` with it, or the stub will
        describe a reduction that could not have happened.
    """

    is_distributed = True
    rank = 0
    world_size = 4

    def all_gather_max_diag(self, value):
        """Return a FIXED (max, argmax, per_rank) so the emit's text is exactly predictable."""
        per_rank = [1.0, 2.0, 4.0, 3.0]
        return 4.0, 2, per_rank

    def all_reduce_max(self, value):
        """The shipped path, which must emit NOTHING."""
        return 4.0


def test_the_rank_diagnostic_is_READ_when_the_flag_is_set(monkeypatch, capsys):
    """With the flag set, `_consensus_max` PRINTS the argmax and the per-rank medians.

    This is the regression it exists for. Before it, `argmax_rank` and `per_rank_ms` were computed
    and DISCARDED on every path -- every reference outside this module lived in this test file, and
    `benchmark/` never mentioned them, so the harness (the only path that runs two trees) threw the
    diagnostic away. A field with no reader is indistinguishable from a field that is never set.
    """
    monkeypatch.setenv(BU.RANK_DIAG_ENV, "1")
    reduced, argmax, per_rank = BU._consensus_max(_DiagStub(), 1.0)
    assert (reduced, argmax, per_rank) == (4.0, 2, [1.0, 2.0, 4.0, 3.0])
    out = capsys.readouterr().out
    assert "[rank-diag]" in out, f"the diagnostic was computed and not printed; stdout was {out!r}"
    assert "argmax_rank=2" in out, f"the emit does not name the argmax rank; stdout was {out!r}"
    assert "per_rank_ms=" in out, (
        f"the emit does not carry the per-rank medians; stdout was {out!r}"
    )


def test_the_rank_diagnostic_emits_NOTHING_on_the_shipped_path(monkeypatch, capsys):
    """Without the flag, the reduction is silent -- so a normal run is byte-identical.

    **This is the half that matters.** "Off by default, so it cannot perturb anything" is an
    argument until something checks it; an emit that leaked onto the shipped path would print once
    per timed cell on every rank, which is both noise and a (small) cost inside a benchmark.
    """
    monkeypatch.delenv(BU.RANK_DIAG_ENV, raising=False)
    reduced, argmax, per_rank = BU._consensus_max(_DiagStub(), 1.0)
    assert (reduced, argmax, per_rank) == (4.0, None, None)
    out = capsys.readouterr().out
    assert "[rank-diag]" not in out, (
        f"the diagnostic emit leaked onto the shipped path with {BU.RANK_DIAG_ENV} unset; "
        f"stdout was {out!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The additive launch-overhead model: measure `C` once, subtract it, stop paying 20x to hide it.
# ─────────────────────────────────────────────────────────────────────────────
def test_the_two_point_solve_recovers_a_KNOWN_injected_per_window_cost(monkeypatch):
    """THE control for :func:`measure_launch_overhead`: inject a known `C`, get it back.

    Everything else about the probe is circular -- a solve that returns SOME number looks identical
    to one that works, and the number it returns is exactly what the gate is about to subtract from
    every measurement. So the model is fed a window whose per-window cost is known by construction
    (`measured(N) = device + C/N`, both terms chosen here) and required to recover `C`.

    No GPU: `time_callable` is replaced by the model itself, which is the point -- this test is about
    the ALGEBRA of the solve, and a real device would add noise that hides an off-by-one in it.
    """
    device_ms, injected_c = 4.0, 0.3

    def fake_time_callable(fn, iters, dev, *, stream=None):
        """Return exactly `device + C/iters`, the model this solve inverts."""
        return device_ms + injected_c / iters

    monkeypatch.setattr(BU, "time_callable", fake_time_callable)
    got = BU.measure_launch_overhead(device=torch.device("cuda:0"), dist=None, rounds=3)
    assert math.isclose(got, injected_c, rel_tol=1e-9), (
        f"the two-point solve returned C={got!r} for an injected C={injected_c!r}; the model is "
        f"measured(N) = device + C/N and the solve is (m[n1]-m[n2]) / (1/n1 - 1/n2)"
    )


def test_the_solve_is_independent_of_the_probe_kernels_own_cost(monkeypatch):
    """`C` must come back the same for a 1 us probe and a 100 ms one.

    This is WHY the probe need not resemble the workload -- its `device` term cancels in the
    difference. If it did not, the probe would have to be built per cell and the whole scheme would
    collapse back into per-cell measurement, which is the thing being removed.
    """
    injected_c = 0.25
    seen = []
    for device_ms in (0.001, 100.0):
        def fake(fn, iters, dev, *, stream=None, _d=device_ms):
            return _d + injected_c / iters

        monkeypatch.setattr(BU, "time_callable", fake)
        seen.append(BU.measure_launch_overhead(device=torch.device("cuda:0"), dist=None, rounds=3))
    assert math.isclose(seen[0], seen[1], rel_tol=1e-9), (
        f"C depended on the probe's own device time: {seen[0]!r} vs {seen[1]!r}"
    )
    assert math.isclose(seen[0], injected_c, rel_tol=1e-9)


def test_a_singular_probe_pair_is_REFUSED_rather_than_dividing_by_zero():
    """Equal `probe_iters` makes `1/n1 - 1/n2` zero; refuse it by name, not by ZeroDivisionError."""
    with pytest.raises(ValueError, match="two distinct positive ints"):
        BU.measure_launch_overhead(device=torch.device("cuda:0"), probe_iters=(4, 4))
    with pytest.raises(ValueError, match="two distinct positive ints"):
        BU.measure_launch_overhead(device=torch.device("cuda:0"), probe_iters=(0, 4))


def test_subtract_overhead_removes_C_over_iters_from_every_window(monkeypatch):
    """The correction is `C / iters`, per window -- not `C`, and not applied once to the median.

    Both halves matter. Subtracting the whole `C` would over-correct by a factor of `iters`, which
    at iters=20 turns a 0.3 ms bias into a 5.7 ms one and is a far larger error than the bias.
    Applying it to the final median instead of each window would leave `raw_ms`/`min_ms`/`std_ms`
    uncorrected on the same result object, so a consumer reading `min_ms` gets an uncorrected number
    beside a corrected one.
    """
    device_ms, injected_c, iters = 4.0, 0.3, 20

    def fake(fn, it, dev, *, stream=None):
        return device_ms + injected_c / it

    monkeypatch.setattr(BU, "time_callable", fake)
    res = BU.benchmark_single(
        lambda: None, rounds=3, warmup=1, iters=iters, device=torch.device("cuda:0"),
        dist=None, mode="event", subtract_overhead=injected_c,
    )
    assert math.isclose(res.median_ms, device_ms, rel_tol=1e-9), (
        f"corrected median {res.median_ms!r} should be the device term {device_ms!r}"
    )
    assert all(math.isclose(v, device_ms, rel_tol=1e-9) for v in res.raw_ms), (
        f"per-window values were not corrected: {res.raw_ms!r}"
    )


def test_subtract_overhead_None_is_byte_identical_to_the_previous_behaviour(monkeypatch):
    """The default must not move ANY existing caller's number.

    Every pin in the repo was harvested through this function with no correction. A default that
    shifted the reading by even a fraction of a percent would retire all of them silently, which is
    exactly the failure mode lowering `iters` produced (6/12 cells over band, worst +20.4%).
    """
    def fake(fn, it, dev, *, stream=None):
        return 4.0 + 0.3 / it

    monkeypatch.setattr(BU, "time_callable", fake)
    kw = dict(rounds=3, warmup=1, iters=20, device=torch.device("cuda:0"), dist=None,
              mode="event")
    a = BU.benchmark_single(lambda: None, **kw)
    b = BU.benchmark_single(lambda: None, subtract_overhead=None, **kw)
    assert a.median_ms == b.median_ms == 4.0 + 0.3 / 20
    assert a.raw_ms == b.raw_ms


def test_subtract_overhead_is_REFUSED_in_device_mode_rather_than_ignored():
    """`mode="device"` runs its own loop with a different window structure.

    Accepting the argument there and silently ignoring it is the dangerous shape: the caller believes
    the reading is overhead-free and it is not, so the number is wrong in the direction that reads as
    a regression. Refusing names the mismatch at the call site.
    """
    with pytest.raises(ValueError, match="mode='event' only"):
        BU.benchmark_single(
            lambda: None, rounds=1, warmup=0, iters=1, device=torch.device("cuda:0"),
            dist=None, mode="device", subtract_overhead=0.3,
        )


# ─────────────────────────────────────────────────────────────────────────────
# benchmark_extrapolated — the per-cell two-point solve that replaces paying iters=20.
# ─────────────────────────────────────────────────────────────────────────────
def test_extrapolation_recovers_the_device_term_from_two_cheap_points(monkeypatch):
    """THE control: feed a window with a known `device` and `C`, get `device` back.

    This is the number the gate will compare against a pin, so a solve that returns SOMETHING
    plausible is indistinguishable from one that works. Fed the model exactly -- both terms chosen
    here -- and required to invert it. No GPU: the subject is the algebra, and device noise would
    mask an off-by-one in the weights.
    """
    device_ms, injected_c = 4.0, 0.3

    def fake(fn, iters, dev, *, stream=None):
        return device_ms + injected_c / iters

    monkeypatch.setattr(BU, "time_callable", fake)
    res = BU.benchmark_extrapolated(
        lambda: None, probe_iters=(1, 4), rounds=3, warmup=1,
        device=torch.device("cuda:0"), dist=None, reduce=None,
    )
    assert math.isclose(res.median_ms, device_ms, rel_tol=1e-9), (
        f"extrapolated {res.median_ms!r}, expected the device term {device_ms!r}"
    )


def test_extrapolation_is_independent_of_which_probe_pair_is_used(monkeypatch):
    """A pair that changed the answer would mean `device + C/N` does not fit, and the scheme is void.

    Checked across pairs that differ in BOTH spacing and magnitude. This is the property that lets
    the pair be chosen on cost/noise grounds alone rather than on accuracy.
    """
    device_ms, injected_c = 7.5, 0.42

    def fake(fn, iters, dev, *, stream=None):
        return device_ms + injected_c / iters

    monkeypatch.setattr(BU, "time_callable", fake)
    got = [
        BU.benchmark_extrapolated(
            lambda: None, probe_iters=p, rounds=3, warmup=1,
            device=torch.device("cuda:0"), dist=None, reduce=None,
        ).median_ms
        for p in ((1, 2), (1, 4), (2, 8), (1, 20))
    ]
    assert all(math.isclose(v, device_ms, rel_tol=1e-9) for v in got), (
        f"the extrapolate depended on the probe pair: {got!r}"
    )


def test_extrapolation_refuses_a_singular_probe_pair():
    """Equal `probe_iters` makes `1/n1 - 1/n2` zero -- refuse by name, not by ZeroDivisionError."""
    with pytest.raises(ValueError, match="two distinct positive ints"):
        BU.benchmark_extrapolated(lambda: None, probe_iters=(4, 4), device=torch.device("cuda:0"))


def test_extrapolation_reports_the_LARGER_probes_raw_rounds_not_invented_ones(monkeypatch):
    """`raw_ms` must describe a real measurement, never a fabricated spread for a derived value.

    An extrapolate has no per-round samples of its own. Synthesising some -- by, say, applying the
    correction to each round of one probe -- would hand a consumer a `std_ms` that looks like the
    extrapolate's uncertainty and is not: the solve amplifies both probes' noise by weights the
    per-round values know nothing about. Reporting the larger probe's raw rounds is honest about
    what was measured, and `iters` says which probe they came from.
    """
    def fake(fn, iters, dev, *, stream=None):
        return 4.0 + 0.3 / iters

    monkeypatch.setattr(BU, "time_callable", fake)
    res = BU.benchmark_extrapolated(
        lambda: None, probe_iters=(1, 4), rounds=5, warmup=0,
        device=torch.device("cuda:0"), dist=None, reduce=None,
    )
    assert res.iters == 4, "iters must name the probe raw_ms came from"
    assert len(res.raw_ms) == 5
    assert all(math.isclose(v, 4.0 + 0.3 / 4, rel_tol=1e-9) for v in res.raw_ms), (
        f"raw_ms should be the n2 probe's UNCORRECTED rounds; got {res.raw_ms!r}"
    )
    assert not math.isclose(res.median_ms, statistics.median(res.raw_ms), rel_tol=1e-9), (
        "median_ms must be the extrapolate, not the median of raw_ms"
    )


def test_both_probes_are_timed_INSIDE_one_round(monkeypatch):
    """The two probes must be adjacent, because the solve is a DIFFERENCE.

    Timing all of probe A and then all of probe B would let a drift that lands between the two loops
    enter the difference, where the solve amplifies it by `n2/(n2-n1)` instead of cancelling it. This
    asserts the interleaving by recording the ORDER of the iters values seen.
    """
    seen = []

    def fake(fn, iters, dev, *, stream=None):
        seen.append(iters)
        return 4.0 + 0.3 / iters

    monkeypatch.setattr(BU, "time_callable", fake)
    BU.benchmark_extrapolated(
        lambda: None, probe_iters=(1, 4), rounds=3, warmup=0,
        device=torch.device("cuda:0"), dist=None, reduce=None,
    )
    assert seen == [1, 4, 1, 4, 1, 4], (
        f"probes must alternate within each round so a drift cancels in the difference; saw {seen!r}"
    )


# ── the host-dispatch reading's own stability check ───────────────────────────────────────────
# MOVED here with `host_dispatch_us` itself (docs/fix_test_setup.md J5); previously in
# `tests/perf/test_calibration.py`. Unchanged apart from the `BU.` prefix and the dropped
# `@matrix_exempt` (this path is out of the audit's scope and the decorator is not imported here --
# see the note above `_DiagStub`).
#
# ONE coverage delta, stated rather than glossed: these two tests script the timer and neuter
# `torch.cuda.synchronize`, so they need no GPU and used to run on a CPU-only box. Here they inherit
# this module's CUDA `pytestmark`. That gate is the FILE's contract, not these tests' requirement,
# and the other ~40 tests of `bench_timing` cannot run on such a box anyway -- so the delta is two
# tests on a venue that was already covering almost none of this module.
def _scripted_perf_counter(monkeypatch, spans):
    """Drive `host_dispatch_us`'s timer from a script of (t0, t1) pairs, one pair per loop count.

    Purpose
        Lets the two loop counts be made to agree or disagree on demand, so the outlier rejection
        can be tested without a GPU and without depending on how busy the machine happens to be --
        which is the very thing that made the untested version of this check flaky.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        spans: Flat sequence of `perf_counter` return values, consumed two at a time. Each of
            `_HOST_N_SAMPLES` interleaved samples consumes FOUR: (t0, t1) for n=50 then (t0, t1) for
            n=200. Too few and the test fails with StopIteration rather than a confusing assertion.

    Returns:
        None; patches `time.perf_counter` and neuters `torch.cuda.synchronize` for the test.
    """
    import time as _time

    import torch as _torch

    monkeypatch.setattr(_torch.cuda, "synchronize", lambda *a, **k: None)
    it = iter(spans)
    monkeypatch.setattr(_time, "perf_counter", lambda: next(it))


def test_one_contaminated_sample_is_medianed_away_rather_than_refusing_the_reading(monkeypatch):
    """A burst in a MINORITY of samples is rejected, and the surviving median is returned.

    The check exists to catch launch-queue backpressure, but each reading is already a MEAN over its
    loop, so per-call jitter is not what trips it -- a rare additive burst is, e.g. an OS
    descheduling event landing inside one loop. Measured on an untouched tree, judging on a single
    pair refused a good reading in 4 of 20 runs, and the fires carried no information about the code
    under test: one happened while timing the bare `GemmSm90` reference.

    Note the value asserted, not just the absence of a raise. Medianing fixes the returned NUMBER as
    well as the pass/fail decision -- the 300 us excursion must not survive into the ratio the
    dispatch pin is compared against.
    """
    _scripted_perf_counter(
        monkeypatch,
        [
            0.0,
            0.005,  # sample 1, n=50:  100 us/call
            0.0,
            0.020,  # sample 1, n=200: 100 us/call
            0.0,
            0.005,  # sample 2, n=50:  100 us/call
            0.0,
            0.060,  # sample 2, n=200: 300 us/call -- the burst, a minority of one
            0.0,
            0.005,  # sample 3, n=50:  100 us/call
            0.0,
            0.020,  # sample 3, n=200: 100 us/call
        ],
    )
    assert BU.host_dispatch_us(lambda: None) == pytest.approx(100.0)


def test_a_disagreement_in_every_sample_still_refuses_the_reading(monkeypatch):
    """Rejecting outliers must not become "eventually accept anything".

    A kernel slow enough to fill the launch queue fills it in EVERY sample, so the disagreement
    survives the median and the reading is still refused. The number it would otherwise return is
    throttled DEVICE time wearing a host cost's clothing -- 243 us/call against the kernel's own
    499 -- and admitting it would make the dispatch pins silently measure the wrong quantity.

    This is the half that a retry-until-agreement scheme gets wrong: three chances to pass lets a
    backpressured kernel through on the one attempt that happens to agree, whereas a median of the
    samples has no such escape.
    """
    _scripted_perf_counter(monkeypatch, [0.0, 0.005, 0.0, 0.060] * BU._HOST_N_SAMPLES)
    with pytest.raises(RuntimeError, match=r"not stable across loop counts"):
        BU.host_dispatch_us(lambda: None)
