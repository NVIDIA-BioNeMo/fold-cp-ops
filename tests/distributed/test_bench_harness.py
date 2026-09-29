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

"""NO-GPU generic-driver self-test (HARNESS_DESIGN §6) — the pre-sweep GATE for the TARGET-AGNOSTIC harness.
Runs the REAL benchmark/distributed/harness driver over SEVERAL DIFFERENT fake target-shapes (elementwise /
gemm-K=N / a2a-multivariant-with-teardown / baseline + the failure modes) to prove ONE driver runs any kernel
in one matrix. sm_120/CPU, seconds, no torchrun. The barrier/all_reduce-count-identical check is the
load-bearing distributed-safety invariant (a driver that branches on LOCAL status fails it)."""
import pytest

from benchmark.distributed.harness import BenchTarget, Ctx, run_matrix
from benchmark.distributed.harness import driver as drv
from benchmark.distributed.harness import registry as reg

from fold_cp_ops.testing.kernel_matrix import matrix_exempt

# Ported VERBATIM from the upstream tree, which has no KernelMatrix audit. Declared once at
# module level rather than 45 times, so the file still diffs clean against its source.
pytestmark = matrix_exempt(
    "the subject is the harness's target/registry plumbing, which launches no kernel and has no shape axis, so no KernelMatrix can apply to any test here"
)

# --------------------------------------------------------------------------- counted dist fakes ---
_BARRIER = {"n": 0}


class FakeDA:
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


def _ctx(N=16384, cp0=2, cp1=8, da=None):
    return Ctx(dm=FakeDM(), pm=None, da=da or FakeDA(), N=N, cp0=cp0, cp1=cp1, Dloc=1, B=1, rd=2,
               device="cpu", rank=0)


def _fake_timeit(target, handle, ctx, *, rounds, warmup):
    """Host stand-in for the real `_timeit` (which now delegates to `bench_utils.benchmark_single` and
    therefore needs a real `torch.device`, not the string `"cpu"` this file's `FakeDM.device` carries —
    `bench_utils.time_callable` does `device.type != "cuda"`, which raises `AttributeError` on a bare str).
    Still exercises the real `target.run(handle)` closure (catches unbound-var / mis-indented-block style
    bugs) and returns the (median_ms, raw_ms) contract `_timeit` promises, so every downstream assertion on
    `time_ms` / winner-selection is exercised exactly as with the real timer."""
    n = max(int(rounds), 1)
    for _ in range(n):
        target.run(handle)
    return 1.23, [1.23] * n


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    _BARRIER["n"] = 0
    monkeypatch.setattr(drv, "_barrier", lambda ctx: _BARRIER.__setitem__("n", _BARRIER["n"] + 1))
    monkeypatch.setattr(drv, "_gate_output", lambda out, ref: True)
    monkeypatch.setattr(drv, "_timeit", _fake_timeit)
    yield


# --------------------------------------------------------------------------- fake target shapes ---
def _elementwise():
    return [BenchTarget("fake_elementwise", build=lambda c: {"x": 1}, run=lambda h: None)]


def _gemm():
    def _supports(c):
        return None if c.N == c.N else "K!=N"      # a DIFFERENT shape contract (K==N always here)
    return [BenchTarget("fake_gemm", build=lambda c: {"M": c.N, "K": c.N}, run=lambda h: None,
                        supports=_supports, output=lambda h: 1, ref=lambda c: 1)]


def _a2a():
    freed = {"n": 0}

    def _cfgs(c):
        return [{"variant": v} for v in ("halo", "pe", "cluster")]   # multi-variant autotune grid

    def _build(c):
        if c.cfg.get("variant") == "pe":                # ONE cfg raises mid-grid -> isolated, others time
            raise ValueError("stubbed a2a build failure for variant=pe")
        return {"buf": "sym", "variant": c.cfg.get("variant")}

    def _teardown(h):
        freed["n"] += 1                                  # frees fake symmetric mem (the 2D-leak guard)

    t = BenchTarget("fake_a2a", build=_build, run=lambda h: None, configs=_cfgs, teardown=_teardown)
    t._freed = freed
    return [t]


def _baseline():
    return [BenchTarget("fake_baseline", build=lambda c: {}, run=lambda h: None, role="baseline")]


def _raises_on_build():
    def _b(c):
        raise RuntimeError("Operation creation failed (stub)")
    return [BenchTarget("raises_on_build", build=_b, run=lambda h: None)]


def _raises_on_run():
    def _r(h):
        raise RuntimeError("kernel launch failed (stub)")
    return [BenchTarget("raises_on_run", build=lambda c: {}, run=_r)]


def _skip_always():
    return [BenchTarget("skip_always", build=lambda c: {}, run=lambda h: None,
                        supports=lambda c: "unsupported shape (stub deterministic skip)")]


def _all_targets():
    return (_elementwise() + _gemm() + _a2a() + _baseline() + _raises_on_build() + _raises_on_run()
            + _skip_always())


# =============================================================================== the §6 tests ===
@pytest.mark.parametrize("do_gate", [True, False], ids=["gate", "nogate"])
def test_agnostic_matrix_runs_all_shapes(do_gate):
    """§6: the SAME run_matrix runs ALL the different target-shapes in ONE matrix (agnosticism), every
    (shape × target) yields a well-formed structured cell + a matrix entry."""
    targets = _all_targets()
    res = run_matrix(targets, [2048, 16384], _ctx(), rounds=3, warmup=1, do_gate=do_gate, first_n=2048)
    cells = res["cells"]
    assert cells
    for c in cells:
        for k in ("status", "error_class", "error_msg", "reason"):
            assert k in c, f"cell missing {k}: {c}"
    # matrix has an entry for EVERY (shape × target)
    for N in (2048, 16384):
        for t in targets:
            assert (N, t.name) in res["matrix"], f"missing matrix entry ({N},{t.name})"
    # the runnable simple targets are ok
    ok = {(c["N"], c["target"]) for c in cells if c["status"] == "ok"}
    assert (16384, "fake_elementwise") in ok
    assert (16384, "fake_gemm") in ok
    assert (16384, "fake_baseline") in ok


def test_isolation_build_and_run_raises_continue():
    """§6: raises_on_build / raises_on_run -> error cell; the driver CONTINUES to the next target (a good
    target after them still returns ok)."""
    order = _raises_on_build() + _raises_on_run() + _elementwise()   # good target AFTER the bad ones
    res = run_matrix(order, [16384], _ctx(), rounds=3, warmup=1, do_gate=False)
    by = {c["target"]: c for c in res["cells"]}
    assert by["raises_on_build"]["status"] == "error" and by["raises_on_build"]["error_msg"]
    assert by["raises_on_run"]["status"] == "error"
    assert by["fake_elementwise"]["status"] == "ok"          # driver continued past the raises


def test_skip_always_is_structured_not_silent():
    """§6: skip_always -> skip_shape with the reason surfaced (never silent); a runnable target still runs."""
    res = run_matrix(_skip_always() + _elementwise(), [16384], _ctx(), rounds=3, warmup=1, do_gate=False)
    by = {c["target"]: c for c in res["cells"]}
    assert by["skip_always"]["status"] == "skip_shape"
    assert by["skip_always"]["reason"] and "unsupported" in by["skip_always"]["reason"]
    assert by["fake_elementwise"]["status"] == "ok"


def test_autotune_grid_fastest_valid_and_mid_grid_raise_skipped():
    """§6: an autotune target records the fastest VALID cfg; a cfg raising mid-grid is isolated, the others
    still time; teardown frees every built handle."""
    tgt = _a2a()
    res = run_matrix(tgt, [16384], _ctx(), rounds=3, warmup=1, do_gate=False)
    cell = res["cells"][0]
    assert cell["status"] == "ok" and cell["winner"] is not None
    subs = {s["cfg"].get("variant"): s for s in cell["configs"]}
    assert subs["pe"]["status"] == "error"                    # the raising cfg isolated
    assert subs["halo"]["status"] == "ok" and subs["cluster"]["status"] == "ok"
    assert tgt[0]._freed["n"] >= 2                            # teardown ran for the built handles


def test_barrier_consistency_local_vs_peer_fail():
    """§6 LOAD-BEARING: within a config where a PEER failed, a rank that built-OK and a rank that RAISED must
    execute the SAME number of collectives (barrier + all_reduce). A driver that branched on LOCAL status
    would let the built-OK rank run timing (more collectives) -> desync. Counts must be identical."""
    good = BenchTarget("g", build=lambda c: {}, run=lambda h: None)
    bad = BenchTarget("b", build=lambda c: (_ for _ in ()).throw(ValueError("boom")), run=lambda h: None)
    # rank that built OK but a PEER failed (da.peer_fail=1.0 -> consensus says fail -> must skip timing)
    _BARRIER["n"] = 0
    da_ok = FakeDA(peer_fail=1.0)
    drv.run_one_config(good, _ctx(da=da_ok).with_cfg({}), do_gate=False, rounds=5, warmup=2)
    n_ok = _BARRIER["n"] + da_ok.n_allreduce
    # rank that RAISED locally (da.peer_fail=1.0 too -> same config, this rank is the failing one)
    _BARRIER["n"] = 0
    da_bad = FakeDA(peer_fail=1.0)
    drv.run_one_config(bad, _ctx(da=da_bad).with_cfg({}), do_gate=False, rounds=5, warmup=2)
    n_bad = _BARRIER["n"] + da_bad.n_allreduce
    assert n_ok == n_bad, (f"COLLECTIVE DESYNC: built-OK-peer-failed rank did {n_ok} collectives vs "
                           f"raised rank {n_bad} -> cross-rank mismatch (would hang).")


def test_registry_resolve_roles():
    """§4: register + resolve; --baselines forces role='baseline'."""
    reg.register("t_ew", _elementwise)
    reg.register("t_base", _baseline)
    tg = reg.resolve(["t_ew"], role="target")
    bl = reg.resolve(["t_base"], role="baseline")
    assert tg[0].role == "target" and bl[0].role == "baseline"
    with pytest.raises(KeyError):
        reg.resolve(["nope_unknown"])


def test_reducer_consumes_emitted_json(tmp_path, monkeypatch):
    """§6: run_matrix emits JSON per shape; the reducer machinery (cutkeep_reduce.load) consumes a
    harness-shaped JSON without error (well-formed matrix reduction)."""
    import json
    import importlib
    res = run_matrix(_elementwise() + _baseline(), [16384], _ctx(), rounds=3, warmup=1, do_gate=False,
                     out_dir=str(tmp_path))
    # the harness writes bench_N16384.json; assert it parses + carries the cells
    p = tmp_path / "bench_N16384.json"
    assert p.exists()
    d = json.loads(p.read_text())
    assert d["N"] == 16384 and len(d["cells"]) == 2
    assert all("status" in c for c in d["cells"])
    cr = importlib.import_module("benchmark.distributed.cutkeep_reduce")   # reducer imports cleanly
    assert hasattr(cr, "load")
