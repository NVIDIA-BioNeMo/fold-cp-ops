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

"""cluster_drain adapter (HARNESS_DESIGN §7.2) — adapter #1. The PRODUCTION back-A2A cluster_drain variants
+ the 2kernel baseline as BenchTargets, wrapping the EXISTING back_a2a_store_bench builders
(_build_cluster, _build_inputs_2d_kn, _reshard_2d, _cluster_forced, the 16-B shape gate, the
tile×cluster_n grid). NO kernel logic is duplicated here — this is a thin plug-in. The keep sweep is:
  python -m benchmark.distributed.harness --targets cluster,cluster_multislot
      --baselines 2kernel ...

NOTE: the old coalesce / differential / strided_putwarp variants were REMOVED at the kernel source
(gemm_sm90_a2a.py: "IB-drain variant is REMOVED at the source"; configure_a2a_gemm_native dropped the
coalesce/differential/ib_reap kwargs) — only the cluster_drain family (cluster_drain=True, cluster_n=…)
is production-reachable, so this adapter exposes only those + the 2kernel baseline.
"""

import os
import sys
import traceback

import benchmark.distributed.back_a2a_store_bench as rp
from benchmark.distributed.harness.registry import register
from benchmark.distributed.harness.target import BenchTarget

# "cluster_roundpark" is NOT offered, and the docstring above already said so ("this adapter exposes
# only those + the 2kernel baseline") while the tuple contradicted it. A later change removed the
# kwarg from configure_a2a_gemm_native, so the target could only ever raise TypeError -- measured
# 2026-08-20 at 164 TypeErrors in one sweep. See back_a2a_store_bench._cluster_forced.
_FUSED = ("cluster", "cluster_multislot")


def _supports(ctx):
    """The ONLY shape constraint: the 16-B TMA alignment (deterministic on every rank). Reuses rp._shape_check."""
    ok, reason = rp._shape_check(ctx.N, ctx.cp0, ctx.cp1)
    return None if ok else reason


#: Default autotune grid, unchanged: the shipped 3-point tile grid x cluster_n {1,2,4,8}.
_DEFAULT_CLUSTER_NS = (1, 2, 4, 8)


def _scoped_tile_grid():
    """The per-cell tile grid, narrowable by ``CPO_HARNESS_TILE_GRID``.

    Purpose
        Make the ONE intractable region of the sweep reachable. `cluster` enumerates
        ``cluster_n{1,2,4,8} x completions x AUTOTUNE_TILE_GRID(3)`` = 12 COLD compiles per cell
        (the harness runs one fresh process per cell and the disk cache is off), and at a straddle
        ``N_token`` each compile hits the ~195 s bitcode cliff that ``rp._parse_tile_grid`` already
        documents. Measured on venue A: 2-node x aligned N is 54 s median and 108/124 ok, 1-node x
        off-grid is ~90 s -- but 2-node x OFF-GRID is 600 s median with **0 of 5 cells completing**,
        at both a 420 s and a 900 s budget, ours and main stalling within one second of each other.
        12 x 195 s is 40 min against any budget worth setting; the arithmetic, not a mystery.

    Semantics
        Unset -> the shipped ``AUTOTUNE_TILE_GRID``, so every existing sweep is byte-identical.
        Set -> parsed by ``rp._parse_tile_grid``, the SAME parser the bench CLI uses, so a spec means
        the same thing in both places rather than through a second implementation.

    Returns:
        A list of ``(tile_m, tile_n, pingpong)`` triples.

    Raises:
        ValueError: From ``rp._parse_tile_grid``, on a malformed spec. Deliberately NOT caught: a
            silent fallback to the full grid would restore the 40 min/cell cost while the operator
            believed the sweep was scoped, which is the failure this exists to prevent.
    """
    spec = os.environ.get("CPO_HARNESS_TILE_GRID", "").strip()
    if not spec:
        return list(rp.AUTOTUNE_TILE_GRID)
    return rp._parse_tile_grid(spec)


def _scoped_cluster_ns():
    """The per-cell ``cluster_n`` pool, narrowable by ``CPO_HARNESS_CLUSTER_NS``.

    Purpose
        The other multiplier on the same 12-compile grid. Scoping the tiles alone cuts 12 to 4;
        scoping both is what makes an off-grid cross-node cell fit inside a cell budget.

    Semantics
        Unset -> ``(1, 2, 4, 8)``, today's pool, so existing sweeps do not move. Set -> a
        comma-separated list of positive ints. ``_variant_cluster_ns`` still collapses this to
        ``[1]`` for non-cluster families, so a scope cannot smuggle a cluster_n into a variant that
        has no such knob.

    Returns:
        A list of ints.

    Raises:
        ValueError: On a non-integer or non-positive entry. Not caught, for the same reason as
            :func:`_scoped_tile_grid`: an operator who mistypes a scope must find out immediately,
            not discover a full-cost sweep hours later.
    """
    spec = os.environ.get("CPO_HARNESS_CLUSTER_NS", "").strip()
    if not spec:
        return list(_DEFAULT_CLUSTER_NS)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)  # ValueError on a non-integer, by design
        if n < 1:
            raise ValueError(f"CPO_HARNESS_CLUSTER_NS entry {n!r} is not positive")
        out.append(n)
    if not out:
        raise ValueError(f"CPO_HARNESS_CLUSTER_NS={spec!r} parsed to an EMPTY pool")
    return out


def _configs_for(variant):
    """The per-cell autotune grid for a variant: AUTOTUNE_TILE_GRID × cluster_n (cluster fam) × completion
    (coalesce/differential). Deterministic on every rank (shape/config math only)."""

    def _cfgs(ctx):
        cns = rp._variant_cluster_ns(variant, _scoped_cluster_ns())
        comps = rp._variant_completions(variant)
        grid = (
            [(128, 128, False)] if rp._variant_family(variant) == "2kernel" else _scoped_tile_grid()
        )
        return [
            {"tile_m": tm, "tile_n": tn, "pingpong": pp, "cluster_n": cn, "completion": comp}
            for cn in cns
            for comp in comps
            for (tm, tn, pp) in grid
        ]

    return _cfgs


def _build_for(variant):
    """build(ctx): alloc recv + ring/stage, call the right rp._build_* at the cfg's tile/cluster_n/completion,
    return an opaque handle. MAY raise (isolated + consensus'd by the driver)."""
    fam = rp._variant_family(variant)
    force_cluster = rp._VARIANT_SPEC.get(variant, (None, None, False))[1]
    differential = rp._VARIANT_SPEC.get(variant, (None, None, False))[2]

    def _build(ctx):
        import torch

        from_dlpack = (
            rp.from_dlpack
        )  # via the module attr so the no-GPU self-test's monkeypatch applies
        cfg = ctx.cfg
        pp = bool(cfg.get("pingpong", False))
        # FIX-2 PP1 GATE (defense-in-depth): _configs_for sources AUTOTUNE_TILE_GRID (no pp1 entries) so
        # this never fires today, but a future --tile-grid-style override must not silently reach
        # _build_cluster/_build_coalesce at pingpong=True (see back_a2a_store_bench._pp_disallowed /
        # run_cell's PP1 GATE for the deadlock / wrong-answer mechanism). MAY raise (isolated by driver.py).
        if pp and rp._pp_disallowed(fam):
            raise ValueError(
                f"pingpong=True gated for {variant!r} (cluster/coalesce families; see "
                "back_a2a_store_bench._pp_disallowed)"
            )
        rp.TILE = (
            int(cfg["tile_m"]),
            int(cfg["tile_n"]),
        )  # the sweep's module-global tile (per-config)
        cluster_n = int(cfg.get("cluster_n", 1))
        reap = cfg.get("completion") == "ib_reap"
        cp0, cp1, N = ctx.cp0, ctx.cp1, ctx.N
        dm, pm = ctx.dm, ctx.pm
        device = ctx.device
        grid_ctas = rp.get_max_active_clusters(1)
        rd = ctx.rd
        N_loc, N_j_loc = N // cp0, N // cp1
        pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
        pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()
        pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)
        A, Bt, recv, (cA, cB, cD) = rp._build_inputs_2d_kn(
            device, dm.rank, N, cp0, cp1, ctx.Dloc, ctx.B
        )
        h = {
            "recv": recv,
            "stage_buf": None,
            "compiled": None,
            "run_args": None,
            "reshard": None,
            "A": A,
            "Bt": Bt,
            "pe_dev_t": pe_dev_t,
        }
        try:
            if fam == "2kernel":
                h["reshard"] = rp._reshard_2d(
                    pm,
                    cp0=cp0,
                    cp1=cp1,
                    Dloc=ctx.Dloc,
                    B=ctx.B,
                    N_i_loc=N_loc,
                    N_j_loc=N_j_loc,
                    device=device,
                )
                return h
            if fam == "coalesce":
                ring = rp._symmetric_empty(
                    (grid_ctas, rd, cp1, rp.TILE[0], N_j_loc), torch.bfloat16, device
                )
                ring.zero_()
                h["stage_buf"] = ring
                rv = from_dlpack(ring, assumed_align=16).mark_layout_dynamic()
                h["compiled"], h["run_args"], _ = rp._build_coalesce(
                    A,
                    Bt,
                    cA,
                    cB,
                    cD,
                    recv,
                    rv,
                    pe_dev_c,
                    cp=cp0 * cp1,
                    cp0=cp0,
                    cp1=cp1,
                    my_cp_rank=int(pm.my_cp_rank),
                    B=ctx.B,
                    N_loc=N_loc,
                    pe_table=pe_table,
                    N=N,
                    rd=rd,
                    pingpong=pp,
                    reap=reap,
                    differential=differential,
                    lws=8,
                )
            elif fam == "strided":
                ring = rp._symmetric_empty((grid_ctas, rd, 128, rp.TILE[1]), torch.bfloat16, device)
                ring.zero_()
                h["stage_buf"] = ring
                rv = from_dlpack(ring, assumed_align=16).mark_layout_dynamic()
                h["compiled"], h["run_args"], _ = rp._build_strided_putwarp(
                    A,
                    Bt,
                    cA,
                    cB,
                    cD,
                    recv,
                    rv,
                    pe_dev_c,
                    cp=cp0 * cp1,
                    cp0=cp0,
                    cp1=cp1,
                    my_cp_rank=int(pm.my_cp_rank),
                    B=ctx.B,
                    N_loc=N_loc,
                    pe_table=pe_table,
                    N=N,
                    rd=rd,
                    pingpong=pp,
                )
            else:  # cluster family
                nt_j_pp = (N_j_loc + rp.TILE[1] - 1) // rp.TILE[1]
                n_clusters = rp.get_max_active_clusters(cluster_n)
                shape, cfg_kw, _ = rp._cluster_forced(
                    force_cluster, cluster_n, nt_j_pp, cp1, N_j_loc, n_clusters, rd
                )
                stage = rp._symmetric_empty(shape, torch.bfloat16, device)
                stage.zero_()
                h["stage_buf"] = stage
                sv = from_dlpack(stage, assumed_align=16).mark_layout_dynamic()
                h["compiled"], h["run_args"], _ = rp._build_cluster(
                    A,
                    Bt,
                    cA,
                    cB,
                    cD,
                    recv,
                    sv,
                    pe_dev_c,
                    cluster_n=cluster_n,
                    cfg_kw=cfg_kw,
                    cp=cp0 * cp1,
                    cp0=cp0,
                    cp1=cp1,
                    my_cp_rank=int(pm.my_cp_rank),
                    B=ctx.B,
                    N_loc=N_loc,
                    pe_table=pe_table,
                    N=N,
                    rd=rd,
                    pingpong=pp,
                )
            return h
        except Exception:
            _teardown(
                h
            )  # free partial recv/ring/compiled on ANY build failure (strided's ib_ring reject,
            # ptxas register-spill, OOM) -> no symmetric-heap leak accumulating across the sweep.
            #
            # AND null THIS FRAME's own bindings, which `_teardown` cannot reach. `main` frees the
            # symmetric buffers here (`nvshmem_torch.free_tensor(recv)`), which returns the heap
            # REGARDLESS of what still references the tensor; this tree releases by dropping the
            # LAST reference instead, so any surviving alias keeps the block. The traceback of the
            # exception about to be re-raised pins this very frame, so `recv`/`cD`/`stage`/`sv`
            # stay live -- past `run_one_config`'s return, because `err = e` plus that traceback
            # form a reference CYCLE that only the CYCLIC collector breaks, at a moment each rank
            # picks for itself.
            #
            # Measured on 16 ranks (venue E, cp=16 N=12288 D=512, `cluster`), per-config census:
            #   boundary#10  alloc=0.00G   SYM_segments=4     (config 9 SUCCEEDED)
            #   boundary#11  alloc=27.39G  SYM_segments=4     (config 10 RAISED -> A+Bt+recv pinned)
            #   boundary#12  alloc=55.16G  SYM_segments=5     (config 11 RAISED -> a SECOND 9.00 GiB
            #                                                  recv segment had to be CUT)
            # Cutting a segment in a symmetric MemPool is a COLLECTIVE `nvshmem_malloc`, so leaving
            # the release to the collector makes a collective's occurrence depend on GC timing --
            # and rank 0 alone does the printing and the incremental JSON flush, so its collector
            # does not fire where its peers' do. Nulling here restores `main`'s determinism: the
            # symmetric block is returned by refcount, on every rank, at the same statement.
            #
            # A/Bt are deliberately NOT nulled: `main` pins them too (its `free_tensor` covers only
            # recv and stage_buf), so leaving them is parity, and removing them would be an
            # unrelated improvement smuggled into a bring-back fix.
            #
            # Only two of `ring`/`rv`/`stage`/`sv` are bound on any one call (one family per call)
            # and the other two are never bound at all. That needs no pre-initialization and cannot
            # raise: ASSIGNING to a local binds it, and only READING an unbound local raises
            # UnboundLocalError.
            recv = cD = stage = sv = ring = rv = None
            # THIS FRAME IS NOT THE ONLY ONE HOLDING THEM. `recv`, `cD` and the staging view are
            # PARAMETERS of `rp._build_cluster` / `_build_coalesce` / `_build_strided_putwarp`,
            # whose frames are in the same traceback, so nulling here alone still leaves the block
            # live -- which is what `test_a_failed_build_releases_its_symmetric_buffers_without_
            # the_cyclic_collector` fails on if this line is removed. `clear_frames` drops the
            # locals of every frame in the traceback; it SKIPS this one (a frame that is executing
            # cannot be cleared, which is why the explicit nulls above are still required) and
            # clears the deeper ones. Safe for the caller: `driver._err_fields` keeps only
            # `type(exc).__name__` and `repr(exc)`, neither of which reads a frame local.
            traceback.clear_frames(sys.exc_info()[2])
            raise

    return _build


def _run_for(variant):
    fam = rp._variant_family(variant)
    if fam == "2kernel":
        import torch

        def _run(h):
            tri = torch.bmm(h["A"], h["Bt"].transpose(-1, -2))
            h["reshard"](tri)

        return _run

    def _run(h):
        h["compiled"](*h["run_args"])  # the fused drain kernel (the TIMED hot call)

    return _run


def _teardown(h):
    """Release the compiled artifact; DROP the symmetric buffers rather than freeing them.

    `stage_buf`/`recv` come from `rp._symmetric_empty`, i.e. the recycling symmetric MemPool, so
    there is no per-tensor free: clearing the handle returns the block to the pool. Calling
    nvshmem4py's `free_tensor` here would be the wrong allocator AND would place a COLLECTIVE
    free on the object-destruction path, where a GC-order difference across ranks deadlocks.

    Input requirements:
        h: the build handle (or a partial one -- this runs on the build-failure path too), so
           every key is fetched with `.get` and a missing key is not an error.
    """
    compiled = h.get(
        "compiled"
    )  # free the compiled kernel too (mirrors back_a2a_store_bench:709)
    if compiled is not None:
        try:
            compiled.free()
        except Exception:
            pass
    for k in ("stage_buf", "recv"):
        h[k] = None


def _factory():
    ts = []
    for v in _FUSED:
        ts.append(
            BenchTarget(
                v,
                build=_build_for(v),
                run=_run_for(v),
                role="target",
                supports=_supports,
                teardown=_teardown,
                configs=_configs_for(v),
            )
        )
    ts.append(
        BenchTarget(
            "2kernel",
            build=_build_for("2kernel"),
            run=_run_for("2kernel"),
            role="baseline",
            supports=_supports,
            teardown=_teardown,
            configs=_configs_for("2kernel"),
        )
    )
    return ts


# every variant name registers as its own module name (so --targets coalesce,differential works) + a bundle
for _v in _FUSED + ("2kernel",):
    register(_v, (lambda vv: lambda: [t for t in _factory() if t.name == vv])(_v))
register("cluster_drain", _factory)
