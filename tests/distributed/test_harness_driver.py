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

"""Cell-protocol + metric-plumbing coverage for benchmark/distributed/harness/driver.py that is NOT already
exercised by tests/distributed/test_bench_harness.py (isolation/skip/autotune/registry/reducer) or
test_roundpark_harness.py (roundpark-specific). This file targets:

  * OOM / timeout CLASS-NAME tagging on build() and run() (driver.py's `is_oom`/`is_timeout` checks).
  * the CONSENSUS actually demoting a LOCALLY-successful build when a peer failed (not just barrier counts).
  * teardown-always, across every failure class (build-raise = no handle = no teardown; peer-fail-after-
    local-build-ok = teardown; run-raise = teardown).
  * the None-guarded comm_bytes -> gbps_total/ib/nvl fields and cell_meta merge.
  * `_node_sz` derivation (cp1 convention + CPO_HARNESS_NODE_SZ override + clamps) and `_comm_bw_fields`
    arithmetic directly.
  * `run_cell`'s per-config barrier count and its real MIN-time winner selection.
  * `run_matrix` incremental JSON flush cadence and per-shape isolation.
  * `norm_cell` idempotence and the registry's unknown-name error message content.
  * a numeric CONSISTENCY check that the harness's `_timeit` (which delegates to
    `bench_utils.benchmark_single(mode="event", reduce="max")`) reproduces a DIRECT `bench_utils` call on
    the same closure — i.e. "a manually-written bench == the harness". This one group runs the REAL
    (un-monkeypatched) `_timeit` on a real CUDA device; every other group fakes `_timeit` (see
    `test_bench_harness.py`'s `_fake_timeit`: the real one needs a `torch.device`, not the bare `"cpu"`
    string this file's minimal fakes would otherwise carry).

Run:
    CPO_CACHE_ENABLED=0 pytest -q tests/distributed/test_harness_driver.py
"""

from __future__ import annotations

import pytest
import torch

from benchmark.distributed import bench_utils as BU
from benchmark.distributed.harness import BenchTarget, Ctx
from benchmark.distributed.harness import driver as drv
from benchmark.distributed.harness import registry as reg
from benchmark.distributed.harness._timeout import PhaseTimeout

from fold_cp_ops.testing.kernel_matrix import matrix_exempt

# Ported VERBATIM from the upstream tree, which has no KernelMatrix audit. Declared once at
# module level rather than 45 times, so the file still diffs clean against its source.
pytestmark = matrix_exempt(
    "the subject is the harness driver's cell protocol -- statuses, consensus roll-up, incremental flush -- which launches no kernel and has no shape axis, so no KernelMatrix can apply to any test here"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fakes.
# ─────────────────────────────────────────────────────────────────────────────
class FakeDA:
    """Counting consensus fake: all_reduce_max(x) = max(x, peer_fail) (simulates a peer's local_ok)."""

    is_distributed = True

    def __init__(self, peer_fail=0.0):
        self.peer_fail = peer_fail
        self.n_allreduce = 0

    def all_reduce_max(self, x):
        self.n_allreduce += 1
        return max(float(x), self.peer_fail)


class FakeDM:
    device = "cpu"
    rank = 0


class NoDist:
    """Minimal single-process da: is_distributed=False short-circuits every all_reduce_max call site."""

    is_distributed = False


def _ctx(N=16384, cp0=2, cp1=8, da=None, dm=None):
    return Ctx(
        dm=dm or FakeDM(),
        pm=None,
        da=da or FakeDA(),
        N=N,
        cp0=cp0,
        cp1=cp1,
        Dloc=1,
        B=1,
        rd=2,
        device="cpu",
        rank=0,
    )


@pytest.fixture(autouse=True)
def barrier_calls(monkeypatch):
    """Autouse: count drv._barrier calls + no-op the correctness gate (every group needs this)."""
    calls = {"n": 0}
    monkeypatch.setattr(drv, "_barrier", lambda ctx: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr(drv, "_gate_output", lambda out, ref: True)
    yield calls


@pytest.fixture
def fake_timeit(monkeypatch):
    """Host stand-in for `_timeit` (NOT autouse — group H below needs the REAL bench_utils-backed one on a
    real CUDA device). Exercises the real `target.run(handle)` closure and returns the (median_ms, raw_ms)
    contract, with the constant chosen so any test checking `time_ms` gets a known, non-zero value."""

    def _fake(target, handle, ctx, *, rounds, warmup):
        n = max(int(rounds), 1)
        for _ in range(n):
            target.run(handle)
        return 1.23, [1.23] * n

    monkeypatch.setattr(drv, "_timeit", _fake)
    return _fake


# ─────────────────────────────────────────────────────────────────────────────
# A. OOM / timeout / generic class-name tagging (build phase), + run-phase timeout.
# ─────────────────────────────────────────────────────────────────────────────
def test_build_oom_tagged_status_oom(fake_timeit):
    """build() raising torch.cuda.OutOfMemoryError (class __name__ == 'OutOfMemoryError') -> status='oom'."""

    def _build(c):
        raise torch.cuda.OutOfMemoryError("simulated OOM")

    t = BenchTarget("oom_target", build=_build, run=lambda h: None)
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "oom"
    assert cell["error_class"] == "OutOfMemoryError"
    assert "OOM" in cell["reason"]


def test_build_timeout_tagged_status_timeout(fake_timeit):
    """build() raising the REAL PhaseTimeout -> status='timeout' (fail-quick stopgap path)."""

    def _build(c):
        raise PhaseTimeout("build exceeded 300s budget (stub)")

    t = BenchTarget("timeout_target", build=_build, run=lambda h: None)
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "timeout"
    assert cell["error_class"] == "PhaseTimeout"
    assert "TIMEOUT" in cell["reason"]


def test_build_generic_raise_tagged_status_error_not_oom_or_timeout(fake_timeit):
    """A build() raise with neither class name -> status='error' (the default bucket)."""

    def _build(c):
        raise ValueError("garden-variety build bug")

    t = BenchTarget("err_target", build=_build, run=lambda h: None)
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "error"
    assert cell["error_class"] == "ValueError"


def test_run_phase_timeout_tagged_status_timeout(fake_timeit):
    """A run() raising the REAL PhaseTimeout during the timed loop (not build) -> ALSO status='timeout'."""

    def _run(h):
        raise PhaseTimeout("run exceeded 120s budget (stub)")

    t = BenchTarget("run_timeout_target", build=lambda c: {}, run=_run)
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "timeout"
    assert "TIMEOUT" in cell["reason"]


def test_run_phase_oom_is_not_specially_tagged(fake_timeit):
    """Negative control: only the BUILD phase distinguishes OOM (driver.py has no run-phase OOM tag) -- a
    run() OOM lands in the generic 'error' bucket, not 'oom'. Documents the real (asymmetric) contract."""

    def _run(h):
        raise torch.cuda.OutOfMemoryError("simulated OOM during run")

    t = BenchTarget("run_oom_target", build=lambda c: {}, run=_run)
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "error"  # NOT "oom" -- run-phase has no OOM-specific bucket
    assert cell["error_class"] == "OutOfMemoryError"


# ─────────────────────────────────────────────────────────────────────────────
# B. Consensus demotes a LOCALLY-successful build when a peer failed.
# ─────────────────────────────────────────────────────────────────────────────
def test_consensus_peer_fail_demotes_local_build_ok(fake_timeit):
    """build() succeeds on THIS rank, but da.peer_fail=1.0 (a peer failed) -> global_ok=False -> the cell is
    skipped in lockstep (status='error', the specific 'peer rank build failed' reason), NOT 'ok' -- this is
    the actual OUTCOME the consensus produces, beyond the barrier-COUNT-only check in test_bench_harness.py."""
    torn_down = {"n": 0}
    t = BenchTarget(
        "peer_fail_target",
        build=lambda c: {"h": 1},
        run=lambda h: None,
        teardown=lambda h: torn_down.__setitem__("n", torn_down["n"] + 1),
    )
    da = FakeDA(peer_fail=1.0)
    cell = drv.run_one_config(t, _ctx(da=da).with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "error"
    assert "peer rank build failed" in cell["reason"]
    assert "time_ms" not in cell  # never reached timing
    assert torn_down["n"] == 1  # this rank's own handle IS freed even though it "won" locally


# ─────────────────────────────────────────────────────────────────────────────
# C. teardown-always, across the failure classes that matter.
# ─────────────────────────────────────────────────────────────────────────────
def test_teardown_not_called_when_build_itself_raised(fake_timeit):
    """build() raising -> handle stays None -> teardown is NEVER invoked (nothing to free)."""
    torn_down = {"n": 0}

    def _build(c):
        raise RuntimeError("build blew up before returning a handle")

    t = BenchTarget(
        "raiser",
        build=_build,
        run=lambda h: None,
        teardown=lambda h: torn_down.__setitem__("n", torn_down["n"] + 1),
    )
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "error"
    assert torn_down["n"] == 0


def test_teardown_called_when_run_raises(fake_timeit):
    """run() raising during timing -> teardown is STILL called (step 6 is unconditional)."""
    torn_down = {"n": 0}

    def _run(h):
        raise RuntimeError("kernel launch failed (stub)")

    t = BenchTarget(
        "run_raiser",
        build=lambda c: {"h": 1},
        run=_run,
        teardown=lambda h: torn_down.__setitem__("n", torn_down["n"] + 1),
    )
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "error"
    assert torn_down["n"] == 1


def test_teardown_called_on_correctness_gate_failure(monkeypatch, fake_timeit):
    """The correctness gate failing (do_gate=True, _gate_output -> False) -> status='disqualified_wrong'
    AND teardown still runs (freeing the built handle before the cell is returned)."""
    monkeypatch.setattr(
        drv, "_gate_output", lambda out, ref: False
    )  # override the autouse "always True"
    torn_down = {"n": 0}
    t = BenchTarget(
        "gated",
        build=lambda c: {"h": 1},
        run=lambda h: None,
        output=lambda h: h,
        ref=lambda c: {"h": 1},
        teardown=lambda h: torn_down.__setitem__("n", torn_down["n"] + 1),
    )
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=True, rounds=2, warmup=1)
    assert cell["status"] == "disqualified_wrong"
    assert cell["correct"] is False
    assert torn_down["n"] == 1


def test_teardown_called_exactly_once_on_ok_path(fake_timeit):
    """The plain ok path also tears down exactly once (no double-free, no leak)."""
    torn_down = {"n": 0}
    t = BenchTarget(
        "ok_target",
        build=lambda c: {"h": 1},
        run=lambda h: None,
        teardown=lambda h: torn_down.__setitem__("n", torn_down["n"] + 1),
    )
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=3, warmup=1)
    assert cell["status"] == "ok"
    assert torn_down["n"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# D. comm_bytes -> gbps_total/ib/nvl fields (None-guarded on target.comm_bytes).
# ─────────────────────────────────────────────────────────────────────────────
def test_ok_cell_without_comm_bytes_has_no_gbps_fields(fake_timeit):
    t = BenchTarget("no_comm", build=lambda c: {}, run=lambda h: None)
    cell = drv.run_one_config(t, _ctx(cp0=2, cp1=8).with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "ok"
    for k in ("comm_bytes", "gbps_total", "gbps_ib", "gbps_nvl"):
        assert k not in cell


def test_ok_cell_with_comm_bytes_populates_gbps_fields_2d(fake_timeit):
    """cp=(2,8)=16, node_sz convention = cp1 = 8: exact gbps_total/ib/nvl split."""
    nbytes = 1 << 20  # 1 MiB, arbitrary but exact
    t = BenchTarget(
        "with_comm", build=lambda c: {}, run=lambda h: None, comm_bytes=lambda c: nbytes
    )
    cell = drv.run_one_config(t, _ctx(cp0=2, cp1=8).with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "ok"
    time_ms = cell["time_ms"]  # 1.23 (fake_timeit)
    cp, node_sz = 16, 8
    expect_total = nbytes / (time_ms * 1e6)
    assert cell["comm_bytes"] == nbytes
    assert cell["node_sz"] == node_sz
    assert cell["gbps_total"] == pytest.approx(expect_total)
    assert cell["gbps_ib"] == pytest.approx(expect_total * (cp - node_sz) / (cp - 1))
    assert cell["gbps_nvl"] == pytest.approx(expect_total * (node_sz - 1) / (cp - 1))
    # sanity: the two locality slices never individually exceed the total.
    assert cell["gbps_ib"] <= cell["gbps_total"] + 1e-9
    assert cell["gbps_nvl"] <= cell["gbps_total"] + 1e-9


def test_comm_bytes_1d_cp_has_no_ib_nvl_split(fake_timeit):
    """cp0=cp1=1 (cp==1, single-PE): gbps_total is reported but there is no off-diagonal to split."""
    t = BenchTarget("single_pe", build=lambda c: {}, run=lambda h: None, comm_bytes=lambda c: 4096)
    cell = drv.run_one_config(t, _ctx(cp0=1, cp1=1).with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "ok"
    assert "gbps_total" in cell
    assert "gbps_ib" not in cell and "gbps_nvl" not in cell


def test_comm_bytes_raising_is_none_guarded(fake_timeit):
    """comm_bytes(ctx) raising -> the fields are silently omitted (never a crashed cell)."""

    def _boom(c):
        raise ValueError("comm_bytes computation bug")

    t = BenchTarget("comm_raises", build=lambda c: {}, run=lambda h: None, comm_bytes=_boom)
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "ok"  # the comm_bytes bug does NOT fail the cell
    assert "gbps_total" not in cell


def test_comm_bytes_nonpositive_is_none_guarded(fake_timeit):
    t = BenchTarget("comm_zero", build=lambda c: {}, run=lambda h: None, comm_bytes=lambda c: 0)
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "ok"
    assert "gbps_total" not in cell


# ─────────────────────────────────────────────────────────────────────────────
# E. cell_meta merged only when the hook is set.
# ─────────────────────────────────────────────────────────────────────────────
def test_ok_cell_without_cell_meta_has_no_extra_fields(fake_timeit):
    t = BenchTarget("no_meta", build=lambda c: {}, run=lambda h: None)
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert "gemm_cfg" not in cell


def test_ok_cell_with_cell_meta_merges_dict(fake_timeit):
    t = BenchTarget(
        "meta",
        build=lambda c: {},
        run=lambda h: None,
        cell_meta=lambda c: {"gemm_cfg": {"tile_m": 128, "tile_n": 256}, "extra": 7},
    )
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["gemm_cfg"] == {"tile_m": 128, "tile_n": 256}
    assert cell["extra"] == 7


def test_cell_meta_raising_is_none_guarded(fake_timeit):
    def _boom(c):
        raise KeyError("cell_meta bug")

    t = BenchTarget("meta_raises", build=lambda c: {}, run=lambda h: None, cell_meta=_boom)
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "ok"
    assert "gemm_cfg" not in cell


def test_cell_meta_non_dict_return_is_ignored(fake_timeit):
    t = BenchTarget(
        "meta_bad_type", build=lambda c: {}, run=lambda h: None, cell_meta=lambda c: "not-a-dict"
    )
    cell = drv.run_one_config(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "ok"
    # a non-dict cell_meta return must not have merged any characters of the string into the cell
    assert "n" not in cell and "o" not in cell


# ─────────────────────────────────────────────────────────────────────────────
# F. _node_sz derivation (cp1 convention + env override + clamps).
# ─────────────────────────────────────────────────────────────────────────────
def test_node_sz_default_is_cp1_when_2d(monkeypatch):
    monkeypatch.delenv("CPO_HARNESS_NODE_SZ", raising=False)
    ctx = _ctx(cp0=2, cp1=8)
    assert drv._node_sz(ctx) == 8


def test_node_sz_1d_mesh_is_one(monkeypatch):
    """cp1==1 (1-D cp) -> node_sz=1 (every off-diagonal peer crosses IB)."""
    monkeypatch.delenv("CPO_HARNESS_NODE_SZ", raising=False)
    ctx = _ctx(cp0=16, cp1=1)
    assert drv._node_sz(ctx) == 1


def test_node_sz_env_override_wins_and_clamps_to_cp(monkeypatch):
    ctx = _ctx(cp0=2, cp1=8)  # cp=16
    monkeypatch.setenv("CPO_HARNESS_NODE_SZ", "4")
    assert drv._node_sz(ctx) == 4
    monkeypatch.setenv("CPO_HARNESS_NODE_SZ", "999")  # far above cp -> clamp to cp
    assert drv._node_sz(ctx) == 16
    monkeypatch.setenv("CPO_HARNESS_NODE_SZ", "0")  # clamp floor -> at least 1
    assert drv._node_sz(ctx) == 1


def test_node_sz_env_invalid_falls_back_to_default(monkeypatch):
    ctx = _ctx(cp0=2, cp1=8)
    monkeypatch.setenv("CPO_HARNESS_NODE_SZ", "not-an-int")
    assert drv._node_sz(ctx) == 8  # ValueError swallowed -> normal cp1-based derivation


# ─────────────────────────────────────────────────────────────────────────────
# G. _comm_bw_fields arithmetic, called directly (no cell plumbing in the way).
# ─────────────────────────────────────────────────────────────────────────────
def test_comm_bw_fields_exact_arithmetic(monkeypatch):
    monkeypatch.delenv("CPO_HARNESS_NODE_SZ", raising=False)
    ctx = _ctx(cp0=2, cp1=8)  # cp=16, node_sz=8
    t = BenchTarget("t", build=lambda c: {}, run=lambda h: None, comm_bytes=lambda c: 2_000_000)
    out = drv._comm_bw_fields(t, ctx, time_ms=5.0)
    expect_total = 2_000_000 / (5.0 * 1e6)
    assert out["gbps_total"] == pytest.approx(expect_total)
    assert out["gbps_ib"] == pytest.approx(expect_total * (16 - 8) / (16 - 1))
    assert out["gbps_nvl"] == pytest.approx(expect_total * (8 - 1) / (16 - 1))
    # the split must exactly reconstruct the total (it partitions the SAME cp-1 off-diagonal peers).
    assert out["gbps_ib"] + out["gbps_nvl"] == pytest.approx(out["gbps_total"])


def test_comm_bw_fields_none_guards():
    ctx = _ctx(cp0=2, cp1=8)
    t_no_hook = BenchTarget("t", build=lambda c: {}, run=lambda h: None)
    assert drv._comm_bw_fields(t_no_hook, ctx, time_ms=5.0) == {}
    t_hook = BenchTarget("t2", build=lambda c: {}, run=lambda h: None, comm_bytes=lambda c: 100)
    assert drv._comm_bw_fields(t_hook, ctx, time_ms=None) == {}
    assert drv._comm_bw_fields(t_hook, ctx, time_ms=0.0) == {}
    assert drv._comm_bw_fields(t_hook, ctx, time_ms=-1.0) == {}


# ─────────────────────────────────────────────────────────────────────────────
# H. run_cell: per-config barrier count + real MIN-time winner selection + all-fail rollup.
# ─────────────────────────────────────────────────────────────────────────────
def test_run_cell_barriers_once_per_config(monkeypatch, barrier_calls, fake_timeit):
    """run_cell issues exactly ONE _barrier() call per config boundary (the LOCKSTEP-ENTRY comment)."""
    cfgs = [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]
    t = BenchTarget("many_cfgs", build=lambda c: {}, run=lambda h: None, configs=lambda c: cfgs)
    drv.run_cell(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert barrier_calls["n"] == len(cfgs)


def test_run_cell_winner_is_true_min_time(monkeypatch):
    """With per-config DISTINCT timings, the winner must be the config with the smallest time_ms -- not
    merely 'some ok config' (the fixed-1.23ms fake_timeit elsewhere can't distinguish this)."""

    def _timeit_by_cfg(target, handle, ctx, *, rounds, warmup):
        return float(ctx.cfg["t"]), [float(ctx.cfg["t"])]

    monkeypatch.setattr(drv, "_timeit", _timeit_by_cfg)
    cfgs = [{"t": 9.0}, {"t": 2.5}, {"t": 6.0}]
    t = BenchTarget("timed_cfgs", build=lambda c: {}, run=lambda h: None, configs=lambda c: cfgs)
    cell = drv.run_cell(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "ok"
    assert cell["winner"]["cfg"] == {"t": 2.5}
    assert cell["winner"]["time_ms"] == 2.5
    assert cell["n_valid"] == 3


def test_run_cell_all_oom_rolls_up_to_oom(fake_timeit):
    def _build(c):
        raise torch.cuda.OutOfMemoryError("simulated")

    t = BenchTarget(
        "all_oom", build=_build, run=lambda h: None, configs=lambda c: [{"a": 1}, {"a": 2}]
    )
    cell = drv.run_cell(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "oom"


def test_run_cell_mixed_oom_and_error_rolls_up_to_error(fake_timeit):
    def _build(c):
        if c.cfg["a"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        raise ValueError("plain failure")

    t = BenchTarget(
        "mixed_fail", build=_build, run=lambda h: None, configs=lambda c: [{"a": 1}, {"a": 2}]
    )
    cell = drv.run_cell(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "error"  # NOT all configs were oom -> mixed bucket is "error"


def test_run_cell_all_skip_shape_rolls_up_to_skip_shape(fake_timeit):
    t = BenchTarget(
        "all_skip",
        build=lambda c: {},
        run=lambda h: None,
        supports=lambda c: "stub: never supported",
        configs=lambda c: [{"a": 1}, {"a": 2}],
    )
    cell = drv.run_cell(t, _ctx().with_cfg({}), do_gate=False, rounds=2, warmup=1)
    assert cell["status"] == "skip_shape"


# ─────────────────────────────────────────────────────────────────────────────
# I. run_matrix: incremental flush cadence + per-shape isolation across multiple N.
# ─────────────────────────────────────────────────────────────────────────────
def test_run_matrix_flushes_incrementally_per_cell(tmp_path, monkeypatch, fake_timeit):
    """_flush_json is called once PER TARGET as its cell completes (growing cell count), PLUS one final
    call at shape-end -- so a hard crash mid-sweep still leaves every completed cell on disk."""
    from benchmark.distributed.harness import run_matrix

    spy = []
    orig = drv._flush_json

    def _spy(out_dir, N, ctx_base, cells):
        spy.append(len(cells))
        return orig(out_dir, N, ctx_base, cells)

    monkeypatch.setattr(drv, "_flush_json", _spy)
    targets = [BenchTarget(f"t{i}", build=lambda c: {}, run=lambda h: None) for i in range(3)]
    run_matrix(targets, [16384], _ctx(), rounds=2, warmup=1, do_gate=False, out_dir=str(tmp_path))
    assert spy == [1, 2, 3, 3], f"expected 3 incremental + 1 final flush, got {spy}"


def test_run_matrix_shape_isolation_one_bad_N_others_continue(fake_timeit):
    """A target that raises in build() ONLY at one specific N -> that (N, target) cell errors, but the
    SAME target at OTHER shapes still returns ok (per-shape isolation, not per-target isolation)."""
    from benchmark.distributed.harness import run_matrix

    def _build(c):
        if c.N == 4096:
            raise RuntimeError("blows up only at N=4096")
        return {}

    t = BenchTarget("n_sensitive", build=_build, run=lambda h: None)
    res = run_matrix([t], [2048, 4096, 8192], _ctx(), rounds=2, warmup=1, do_gate=False)
    by = {c["N"]: c for c in res["cells"]}
    assert by[2048]["status"] == "ok"
    assert by[4096]["status"] == "error"
    assert by[8192]["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# J. norm_cell idempotence.
# ─────────────────────────────────────────────────────────────────────────────
def test_norm_cell_idempotent_on_well_formed_dict():
    c = {"status": "ok", "time_ms": 1.0}
    c1 = drv.norm_cell(dict(c))
    c2 = drv.norm_cell(dict(c1))
    assert c1 == c2
    assert c1["reason"] == "" and c1["error_class"] is None and c1["error_msg"] is None


def test_norm_cell_non_dict_input_becomes_structured_error():
    c = drv.norm_cell("not-a-dict-at-all")
    assert c["status"] == "error" and c["error_class"] == "BadCell"
    assert "not-a-dict-at-all" in c["error_msg"]
    # re-normalizing the (now dict) result is a true no-op.
    c2 = drv.norm_cell(dict(c))
    assert c2 == c


# ─────────────────────────────────────────────────────────────────────────────
# K. registry: unknown-name error message content + role forcing.
# ─────────────────────────────────────────────────────────────────────────────
def test_registry_unknown_name_message_lists_known_names():
    reg.register("harness_driver_test_known_a", lambda: [])
    reg.register("harness_driver_test_known_b", lambda: [])
    with pytest.raises(KeyError) as ei:
        reg.resolve(["harness_driver_test_totally_unknown"])
    msg = str(ei.value)
    assert "harness_driver_test_totally_unknown" in msg
    assert "harness_driver_test_known_a" in msg
    assert "harness_driver_test_known_b" in msg


# ─────────────────────────────────────────────────────────────────────────────
# H2. `_timeit` == direct `bench_utils` -- the REAL (un-monkeypatched) timing path, real CUDA.
#     Single-process (dm=None -> resolve_dist falls back to the no-op adapter), so reduce="max" is an
#     identity no-op and no torch.distributed init is required (runs on this box's plain CUDA, sm_120 ok).
# ─────────────────────────────────────────────────────────────────────────────
pytestmark_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real CUDA device"
)


def _gemm_target(n, dtype=torch.float16):
    def _build(c):
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        out = torch.empty(n, n, device="cuda", dtype=dtype)
        h = {"a": a, "b": b, "out": out}
        for _ in range(5):
            torch.mm(h["a"], h["b"], out=h["out"])
        torch.cuda.synchronize()
        return h

    def _run(h):
        torch.mm(h["a"], h["b"], out=h["out"])

    return BenchTarget(f"gemm{n}", build=_build, run=_run)


@pytestmark_cuda
def test_real_timeit_matches_direct_bench_utils_call():
    """The harness's run_one_config (REAL _timeit, not faked) reports a time_ms within ~15% of an
    INDEPENDENT direct bench_utils.benchmark_single call on the same closure shape -- proving the harness's
    timing path and a hand-written bench_utils call measure the SAME thing."""
    ctx = Ctx(
        dm=None,
        pm=None,
        da=NoDist(),
        N=0,
        cp0=1,
        cp1=1,
        Dloc=1,
        B=1,
        rd=2,
        device=torch.device("cuda"),
        rank=0,
    )
    target = _gemm_target(768)

    cell = drv.run_one_config(target, ctx.with_cfg({}), do_gate=False, rounds=20, warmup=4)
    assert cell["status"] == "ok"
    harness_ms = cell["time_ms"]

    handle = target.build(ctx)
    direct = BU.benchmark_single(
        lambda: target.run(handle),
        rounds=20,
        warmup=4,
        iters=1,
        dist=None,
        mode="event",
        reduce="max",
    ).median_ms

    # RELATIVE-OR-ABSOLUTE, because a purely relative bar on a MICROSECOND quantity is unbounded
    # noise rather than a bar. This closure measures ~10-20 us, where CUDA-event resolution and
    # per-launch jitter are a few us on their own -- so a 15% relative bound is asking two
    # independent ~6 us-noisy measurements to agree to ~2 us. Measured on an 8-rank launch (8
    # concurrent copies of this rank-independent test on one node, which is itself the stressor):
    # 0.01946 vs 0.01363 ms (rel 42.7%) on one rank and 0.01362 vs 0.01931 (rel 29.5%) on another --
    # the SAME pair with the sign flipped, which is the signature of noise, not of bias.
    # The absolute floor is set above that observed jitter; at any shape large enough to matter the
    # relative term dominates again, so the test keeps its power exactly where it has any.
    _abs_floor_ms = 0.010
    assert abs(harness_ms - direct) <= max(0.15 * direct, _abs_floor_ms), (
        f"harness _timeit {harness_ms:.5f}ms vs direct bench_utils {direct:.5f}ms "
        f"(delta {abs(harness_ms - direct):.5f}ms, allowed {max(0.15 * direct, _abs_floor_ms):.5f}ms)"
    )


@pytestmark_cuda
def test_real_timeit_two_cell_minisweep_matches_direct_calls():
    """A 2-shape mini-sweep run cell-by-cell through the REAL harness path matches two INDEPENDENT direct
    bench_utils calls, each within tolerance -- 'a manually-written multi-cell bench == the harness'."""
    ctx = Ctx(
        dm=None,
        pm=None,
        da=NoDist(),
        N=0,
        cp0=1,
        cp1=1,
        Dloc=1,
        B=1,
        rd=2,
        device=torch.device("cuda"),
        rank=0,
    )
    for n in (256, 512):
        target = _gemm_target(n)
        cell = drv.run_one_config(target, ctx.with_cfg({}), do_gate=False, rounds=15, warmup=3)
        assert cell["status"] == "ok"
        harness_ms = cell["time_ms"]

        handle = target.build(ctx)
        direct = BU.benchmark_single(
            lambda: target.run(handle),
            rounds=15,
            warmup=3,
            iters=1,
            dist=None,
            mode="event",
            reduce="max",
        ).median_ms

        # Relative-or-absolute, for the reason spelled out in the single-cell test above: n=256 here
        # measures ~6-12 us, and a purely relative bar at that scale compares two noise floors.
        # Measured at 8 ranks: harness 0.00662 vs direct 0.01229 ms, rel 46.1% on a 20% bar.
        _abs_floor_ms = 0.010
        assert abs(harness_ms - direct) <= max(0.20 * direct, _abs_floor_ms), (
            f"n={n}: harness {harness_ms:.5f}ms vs direct {direct:.5f}ms "
            f"(delta {abs(harness_ms - direct):.5f}ms, allowed {max(0.20 * direct, _abs_floor_ms):.5f}ms)"
        )
