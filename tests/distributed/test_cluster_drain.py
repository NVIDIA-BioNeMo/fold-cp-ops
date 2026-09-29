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

"""The `cluster_drain` adapter: its CONFIG GRID (that it can be scoped) and its build-failure
BUFFER LIFETIME (that a rejected config's symmetric buffers are released by refcount).

One file per source, so both live here. The second subject is the teardown-hang regression and its
own reasoning sits above the tests at the bottom of the file; what follows is the first.

Why this file exists
    `cluster` enumerates ``cluster_n{1,2,4,8} x completions x AUTOTUNE_TILE_GRID(3)`` = 12 COLD
    compiles per cell: the harness gives every cell a fresh process and the disk cache is off. At a
    straddle ``N_token`` each compile meets the ~195 s bitcode cliff that
    ``back_a2a_store_bench._parse_tile_grid`` already documents, so a cell costs ~40 minutes.
    Measured on venue A, bucketing every cell by node span x grid alignment:

    =========================  ====  ==========  =====================
    bucket                     n     median s    outcomes
    =========================  ====  ==========  =====================
    1-node x aligned            51          38   48 ok
    2-node x aligned           124          54   108 ok, 9 fail, 7 oom
    2-node x OFF-GRID            5         600   **0 ok**
    =========================  ====  ==========  =====================

    Neither factor alone does it -- single-node off-grid runs ~90 s on venue B with the identical
    12-config grid. It is the interaction, and `front` / `front_a2a` at the same off-grid cross-node
    cell measure in 25 s, which localizes the cost to THIS target's grid rather than to the shape.

No GPU is needed here: the config grid is pure shape/config math, evaluated per rank and required
to be deterministic across ranks (the driver relies on that for lockstep).
"""

from __future__ import annotations

import pytest

import benchmark.distributed.back_a2a_store_bench as rp
from benchmark.distributed.harness.targets import cluster_drain as cd

from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.testing.numeric_guard import numeric_exempt

pytestmark = matrix_exempt(
    "two subjects, neither of which has a shape axis and neither of which launches a kernel: the "
    "adapter's CONFIG GRID -- how many (tile, cluster_n) combinations a cell enumerates and whether "
    "an operator can narrow it -- and the BUFFER LIFETIME on its build-failure path, which is a "
    "Python reference-counting property of the adapter and not a property of any tensor's contents"
)


class _Ctx:
    """Minimal stand-in for the harness Ctx: `_configs_for` reads nothing off it.

    Input requirements: none. It exists because the grid callable takes a ctx positionally; giving
    it a real Ctx would drag in a DistributedManager for a function that never looks at one.
    """


@numeric_exempt("counts configuration tuples; there is no computed tensor to compare")
def test_the_default_grid_is_unchanged_by_the_scope_hooks():
    """With neither env var set, the grid is exactly what it has always been: 12 configs.

    This is the direction that matters most. A scoping hook whose DEFAULT quietly narrowed the pool
    would silently change every existing sweep's meaning -- and a perf sweep that measures fewer
    configs than it used to still produces numbers, so nothing would fail.
    """
    cfgs = cd._configs_for("cluster")(_Ctx())
    assert len(cfgs) == 12, f"default cluster grid is {len(cfgs)} configs, expected 12"
    assert sorted({c["cluster_n"] for c in cfgs}) == [1, 2, 4, 8]
    assert sorted({(c["tile_m"], c["tile_n"]) for c in cfgs}) == sorted(
        {(tm, tn) for tm, tn, _pp in rp.AUTOTUNE_TILE_GRID}
    )


@numeric_exempt("counts configuration tuples; there is no computed tensor to compare")
def test_scoping_both_axes_cuts_the_cold_compiles_per_cell(monkeypatch):
    """Scoping tiles and cluster_n narrows the grid multiplicatively -- 12 -> 2.

    12 x ~195 s is the 40 min/cell that leaves an off-grid cross-node cell unmeasurable at any
    budget worth setting; 2 fits.
    """
    monkeypatch.setenv("CPO_HARNESS_TILE_GRID", "128x256")
    monkeypatch.setenv("CPO_HARNESS_CLUSTER_NS", "1,2")
    cfgs = cd._configs_for("cluster")(_Ctx())
    assert len(cfgs) == 2, f"scoped grid is {len(cfgs)} configs, expected 2"
    assert sorted({c["cluster_n"] for c in cfgs}) == [1, 2]
    assert {(c["tile_m"], c["tile_n"]) for c in cfgs} == {(128, 256)}


@numeric_exempt("counts configuration tuples; there is no computed tensor to compare")
def test_a_scope_cannot_smuggle_cluster_n_into_a_variant_without_that_knob(monkeypatch):
    """`2kernel` still collapses to a single cluster_n and a single tile, scope or no scope.

    The scope narrows a pool; it must not WIDEN one, and a baseline that suddenly enumerated four
    cluster_n values would be comparing the fused target against something the baseline cannot do.
    """
    monkeypatch.setenv("CPO_HARNESS_CLUSTER_NS", "1,2,4,8")
    cfgs = cd._configs_for("2kernel")(_Ctx())
    assert sorted({c["cluster_n"] for c in cfgs}) == [1], f"2kernel took a cluster_n scope: {cfgs}"
    assert {(c["tile_m"], c["tile_n"]) for c in cfgs} == {(128, 128)}


@pytest.mark.parametrize("bad", ["nonsense", "128x", "0", "-2", "1,zero"])
@numeric_exempt("asserts a refusal on malformed input; no numerical comparison")
def test_a_malformed_scope_raises_rather_than_falling_back(monkeypatch, bad):
    """A bad spec RAISES; it never silently reverts to the full grid.

    A silent fallback is the dangerous direction: the operator believes the sweep is scoped, the
    cells cost 40 minutes each again, and the only symptom is a sweep that does not finish -- which
    looks exactly like the problem the scope was meant to solve.
    """
    var = "CPO_HARNESS_CLUSTER_NS" if bad in ("0", "-2", "1,zero") else "CPO_HARNESS_TILE_GRID"
    monkeypatch.setenv(var, bad)
    with pytest.raises(ValueError):
        cd._configs_for("cluster")(_Ctx())


# --------------------------------------------------------------------------------------------- #
# The BUILD-FAILURE path's buffer lifetime. Different subject from the config grid above, same
# source file -- one test file per source, so it lives here rather than in a second file.
#
# `main`'s `_teardown` calls `nvshmem_torch.free_tensor(recv)`, which returns the symmetric heap
# REGARDLESS of what still references the tensor. This tree allocates through the recycling
# `DistributedManager.symmetric_mempool` instead, so a block comes back only when the LAST reference
# dies -- and on the failure path the exception's traceback pins `_build`'s frame, whose locals still
# bind `recv`, `cD`, `stage` and `sv`. `_teardown` nulls the DICT entries and cannot reach a frame.
#
# Measured consequence before the fix, venue E, 16 ranks, cp=16 N=12288 D=512, per-config census:
#   after a SUCCEEDING config   alloc=0.00G   SYM_segments=4
#   after ONE raising config    alloc=27.39G  SYM_segments=4     (A + Bt + recv, 9.00 GiB each)
#   after TWO raising configs   alloc=55.16G  SYM_segments=5     (a SECOND 9.00 GiB recv segment
#                                                                 had to be CUT, free_dev 39->12 GiB)
# Cutting a segment in a symmetric MemPool is a COLLECTIVE `nvshmem_malloc`. Leaving the release to
# the CYCLIC collector therefore makes a collective's occurrence depend on GC timing, and rank 0
# alone does the printing and the incremental JSON flush, so its collector does not fire where its
# peers' do.
#
# No GPU, no process group, no CUDA context: the property under test is a Python reference lifetime,
# and testing it against the REAL `_build` is what makes it a regression test rather than a
# restatement. Everything `_build` reaches into `back_a2a_store_bench` for is monkeypatched.
# --------------------------------------------------------------------------------------------- #


class _Sentinel:
    """A stand-in for a device buffer whose LIFETIME is the whole subject.

    Purpose: be weak-referenceable and cheap, so a test can ask "is this released" without holding
    it. Input requirements: none. `zero_()` exists because `_build` calls it on every staging
    buffer it allocates and returns self so it can be used in an expression.
    """

    def __init__(self, tag):
        self.tag = tag

    def zero_(self):
        return self


class _View:
    """A stand-in for a CuTe tensor built over a buffer -- it RETAINS the buffer, as the real one does.

    Purpose: without the retention this test would be vacuous for `cD`/`sv`, which are the aliases
    that actually keep a symmetric block alive after `h["recv"]` has been nulled. Input
    requirements: `buf` is the object being aliased; it is stored, deliberately.
    """

    def __init__(self, buf):
        self.buf = buf

    def mark_layout_dynamic(self):
        return self


class _BuildCtx:
    """The subset of the harness `Ctx` that `_build` actually reads.

    Input requirements: `cfg` must carry tile_m/tile_n/cluster_n/completion (the adapter reads all
    four); `cp0`/`cp1` must divide `N` (`_build` computes `N // cp0`); `pm.cp_pe_table` must be a
    real integer tensor because `_build` calls `.tolist()` and `.to(torch.int32).contiguous()` on it.
    A CPU tensor is fine -- nothing here reaches a device.
    """

    def __init__(self):
        import torch

        self.cfg = {
            "tile_m": 256,
            "tile_n": 128,
            "pingpong": False,
            "cluster_n": 8,
            "completion": "ib_quiet",
        }
        self.cp0, self.cp1, self.N = 16, 1, 12288
        self.Dloc, self.B, self.rd = 32, 1, 2
        self.dm = type("_DM", (), {"rank": 0})()
        self.pm = type("_PM", (), {"cp_pe_table": torch.arange(16), "my_cp_rank": 0})()
        self.device = "cpu"


def _patch_rp(monkeypatch, boxes, *, raise_in_build):
    """Point every `back_a2a_store_bench` entry `_build` uses at a sentinel factory.

    Args:
        monkeypatch: pytest's fixture; every patch is undone at teardown.
        boxes: dict filled with ``{"recv": weakref, "stage": weakref, "A": weakref}`` -- weak, so
            the test itself never keeps the buffers alive and cannot mask the defect.
        raise_in_build: when True `_build_cluster` raises the real cluster_n>=8 rejection, which is
            the production-reachable trigger (`gemm_sm90_a2a.py`'s autotune-knob guard). When False
            the build succeeds, which is the NEGATIVE control.

    Returns:
        None. Raises nothing.
    """
    import weakref

    def _inputs(device, rank, N, cp0, cp1, Dloc, B):
        A, Bt, recv = _Sentinel("A"), _Sentinel("Bt"), _Sentinel("recv")
        boxes["A"] = weakref.ref(A)
        boxes["recv"] = weakref.ref(recv)
        return A, Bt, recv, (_View(A), _View(Bt), _View(recv))

    def _sym(shape, dtype, device):
        stage = _Sentinel("stage")
        boxes["stage"] = weakref.ref(stage)
        return stage

    def _cluster(*a, **k):
        if raise_in_build:
            raise ValueError(
                "cluster_drain cluster_n=8 is an unreasonable autotune hyper-parameter"
            )
        return object(), (), False

    monkeypatch.setattr(rp, "_build_inputs_2d_kn", _inputs)
    monkeypatch.setattr(rp, "_symmetric_empty", _sym)
    monkeypatch.setattr(rp, "_build_cluster", _cluster)
    monkeypatch.setattr(rp, "from_dlpack", lambda t, **k: _View(t))
    monkeypatch.setattr(rp, "get_max_active_clusters", lambda n: 16)
    monkeypatch.setattr(rp, "_cluster_forced", lambda *a, **k: ((16, 2, 128, 12288), {}, "even"))


def _drive_one_config(ctx, boxes, monkeypatch, *, raise_in_build):
    """Call the REAL `_build` the way `driver.run_one_config` does, and return its stringified cell.

    The shape is load-bearing: `run_one_config` binds the exception to a name (`err = e`) that
    outlives the except block, and that name plus the traceback form the reference CYCLE which
    refcounting cannot break. Returning only strings mirrors `_err_fields`, so nothing the caller
    keeps can pin a buffer -- the lifetime is then purely a property of the code under test.
    """
    _patch_rp(monkeypatch, boxes, raise_in_build=raise_in_build)
    handle, err = None, None
    try:
        handle = cd._build_for("cluster")(ctx)
    except Exception as e:  # noqa: BLE001 - mirrors driver.run_one_config exactly
        err = e
    if handle is not None:
        cd._teardown(handle)
    return {
        "status": "error" if err is not None else "ok",
        "error_class": type(err).__name__ if err is not None else None,
    }


@numeric_exempt("asserts a reference LIFETIME, not a computed value; there is no tensor to compare")
def test_a_failed_build_releases_its_symmetric_buffers_without_the_cyclic_collector(monkeypatch):
    """recv and the staging buffer are gone by REFCOUNT once the config returns -- no `gc.collect()`.

    This is the regression test for the teardown hang. Before the fix both were still alive here,
    and stayed alive until the cyclic collector happened to run -- which is what forced a second
    symmetric SEGMENT (a collective `nvshmem_malloc`) at the next config and made that collective's
    occurrence a function of per-rank GC timing.

    `A` is asserted STILL ALIVE on purpose, and it does two jobs. It is the non-vacuity control:
    the pin is genuinely active in this harness, so an assertion that something is released is
    meaningful rather than trivially true. And it is the PARITY marker: `main` frees only recv and
    stage_buf, so `A`/`Bt` stay pinned there too, and freeing them here would be an unrelated
    improvement rather than the bring-back's behaviour.
    """
    import gc

    gc.collect()  # start from a clean cycle state; NOT called again before the assertions
    boxes = {}
    cell = _drive_one_config(_BuildCtx(), boxes, monkeypatch, raise_in_build=True)
    assert cell["error_class"] == "ValueError", f"expected the cluster_n>=8 rejection, got {cell}"
    assert boxes["recv"]() is None, (
        "recv survived the failed build: the exception's traceback still pins _build's frame, so "
        "the symmetric block is released only when the CYCLIC collector runs"
    )
    assert boxes["stage"]() is None, "the staging buffer survived the failed build (same mechanism)"
    assert boxes["A"]() is not None, (
        "A was released too -- either the pin is not active (this test is vacuous) or the fix went "
        "beyond main, which frees only recv and stage_buf"
    )


@numeric_exempt("asserts a reference LIFETIME, not a computed value; there is no tensor to compare")
def test_a_successful_build_releases_its_symmetric_buffers_too(monkeypatch):
    """The NEGATIVE control: on the path that never raises, refcounting already did the job.

    Kept because it is what makes the failing-path assertion informative. If both paths retained,
    the subject would be the teardown contract in general; the defect was specific to the path that
    leaves a traceback behind, and only a passing success arm shows that.
    """
    import gc

    gc.collect()
    boxes = {}
    cell = _drive_one_config(_BuildCtx(), boxes, monkeypatch, raise_in_build=False)
    assert cell["status"] == "ok", f"the non-raising arm did not build: {cell}"
    assert boxes["recv"]() is None, "recv survived a SUCCESSFUL build's teardown"
    assert boxes["stage"]() is None, "the staging buffer survived a SUCCESSFUL build's teardown"
