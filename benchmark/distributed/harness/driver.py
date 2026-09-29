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

"""The driver (HARNESS_DESIGN §2-3). Iterates (shape × target × cfg) -> structured cells, following the
COLLECTIVE-SYMMETRIC cell protocol: every rank executes the identical sequence of collectives regardless of
any per-cell/-config/-target outcome. A failure on ONE rank becomes the SAME outcome on ALL ranks, decided by
an all_reduce, NEVER by local state -> no 3am hang. The core imports zero kernels; targets are passed in.

The GPU/dist seams (_barrier, _gate_output, _timeit) are module-level so the no-GPU self-test monkeypatches
them; the consensus primitive is ctx.da.all_reduce_max (also faked in-test).
"""
import json
import os

from benchmark.distributed.harness._timeout import phase_deadline, budget_s

_STATUS = {"ok", "error", "skip_shape", "oom", "disqualified_wrong", "timeout"}
_MAX_CONFIGS = 0   # 0 = full autotune grid; >0 truncates each target's grid to the first N (smoke=1). Set by __main__.

# FAIL-QUICK per-config timeout budgets (STOPGAP; see _timeout.py). COLLECTIVE by construction: run_cell's
# unconditional config-boundary barrier gives a COMMON T0, so an IDENTICAL budget on every rank fires the
# SIGALRM at ~T0+budget everywhere -> the overrun is consensus'd (symmetric skip), NOT a per-rank desync.
# BUILD budget must be < the (raised) dist-store timeout so the alarm beats the 600s DistStoreError; it must
# be >> a HEALTHY compile so a normal config never trips it. RUN budget bounds a 2-node drain deadlock.
# 0 disables. Env: CPO_HARNESS_BUILD_BUDGET_S (default 300), CPO_HARNESS_RUN_BUDGET_S (default 120).
_BUILD_BUDGET = budget_s("CPO_HARNESS_BUILD_BUDGET_S", 300.0)
_RUN_BUDGET = budget_s("CPO_HARNESS_RUN_BUDGET_S", 120.0)


# --------------------------------------------------------------- GPU/dist seams (monkeypatched in-test) ---
def _barrier(ctx):
    """Symmetric barrier (every rank). Real impl = the nvshmem/dist barrier injected by the runner."""
    fn = getattr(ctx, "_barrier_fn", None)
    if fn is not None:
        fn()


def _timeit(target, handle, ctx, *, rounds, warmup):
    """Timed hot-loop: pre-barrier, warmup, then `rounds` barrier'd calls; each round's time all_reduce_max'd
    to the SLOWEST PE (the real distributed cost). Returns (med_max_ms, samples). Collective-symmetric: only
    reached when global_ok on ALL ranks."""
    from benchmark.distributed import bench_utils
    # Delegate to the shared, tested timer instead of a hand-rolled loop. mode="event"+reduce="max":
    # FIXED-iters, barrier-bracketed CUDA-event window + all_reduce(MAX) slowest-PE consensus. CUDA events
    # measure true DEVICE time (comm + kernels), so V/median is a real BW — the old perf_counter loop timed
    # async host-DISPATCH (V/dt could exceed the physical link, the >beta_ib tell). mode="event" (NOT
    # "device") is the collective-safe path: "device"'s adaptive rep-count runs DIFFERENT iter counts per
    # rank -> in-kernel/NCCL collectives desync (bench_utils contract). iters=1 = one call per timed window.
    res = bench_utils.benchmark_single(
        lambda: target.run(handle), rounds=rounds, warmup=warmup, iters=1,
        dist=ctx.dm, mode="event", reduce="max",
    )
    return res.median_ms, res.raw_ms


def _gate_output(out, ref):
    """Correctness compare -> bool. Real impl injected; default True (perf-only)."""
    return True


# ------------------------------------------------------------------------------ structured cell helpers ---
def _err_fields(exc):
    return {"error_class": type(exc).__name__, "error_msg": repr(exc)[:1500]}


def norm_cell(cell):
    """Ensure {status, error_class, error_msg, reason} on a cell dict (idempotent, never raises)."""
    if not isinstance(cell, dict):
        return {"status": "error", "error_class": "BadCell", "error_msg": repr(cell)[:200], "reason": ""}
    cell.setdefault("status", "ok")
    cell.setdefault("reason", "")
    cell.setdefault("error_class", None)
    cell.setdefault("error_msg", None)
    return cell


def _consensus_ok(ctx, local_ok):
    """global_ok = (no rank failed). all_reduce_max of the fail flag (0/1); reached by EVERY rank whether it
    built or raised -> the collective itself never desyncs."""
    fail = 0.0 if local_ok else 1.0
    if getattr(ctx.da, "is_distributed", False):
        fail = float(ctx.da.all_reduce_max(fail))
    return fail == 0.0


# ------------------------------------------------------------------------------- the §3 cell protocol -----
def run_one_config(target, ctx, *, do_gate, rounds, warmup):
    """§3: run ONE (target, cfg) with the collective-symmetric protocol. Returns a structured cell dict."""
    base = {"target": target.name, "role": target.role, "cfg": dict(ctx.cfg), "N": ctx.N,
            "cp0": ctx.cp0, "cp1": ctx.cp1, "Dloc": ctx.Dloc}
    # 1. supports() -- DETERMINISTIC on all ranks -> no collective, same decision everywhere
    try:
        reason = target.supports(ctx)
    except Exception as e:
        return norm_cell({**base, "status": "error", "reason": "supports() raised", **_err_fields(e)})
    if reason is not None:
        return norm_cell({**base, "status": "skip_shape", "reason": str(reason)})
    # 2. build (isolated) -> local_ok
    handle = None
    local_ok = True
    err = None
    try:
        # STOPGAP fail-quick: bound the COLD compile so a cluster-family blowup can't wedge the barrier
        # (48-min freeze) or blow the 600s NCCL store timeout (desync crash). PhaseTimeout is an Exception
        # -> caught here like any build failure -> the adapter's own except frees partial symmetric allocs
        # (roundpark._build's `except: _teardown(h); raise`) -> consensus'd symmetric skip below.
        with phase_deadline(_BUILD_BUDGET, "build", N=ctx.N, cfg=dict(ctx.cfg)):
            handle = target.build(ctx)
    except Exception as e:  # ANY build failure isolates (OOM / TIMEOUT tagged below)
        local_ok = False
        err = e
    is_oom = err is not None and type(err).__name__ == "OutOfMemoryError"
    is_timeout = err is not None and type(err).__name__ == "PhaseTimeout"
    # 3. CONSENSUS -> global_ok. If ANY rank failed, ALL skip timing (symmetric).
    global_ok = _consensus_ok(ctx, local_ok)
    if not global_ok:
        if handle is not None and target.teardown is not None:
            try:
                target.teardown(handle)
            except Exception:
                pass
        if err is not None:
            _st = "oom" if is_oom else ("timeout" if is_timeout else "error")
            _rs = ("build OOM (isolated)" if is_oom else
                   ("build TIMEOUT (fail-quick stopgap; sweep continues; REAL fix = <=5s cold compile)"
                    if is_timeout else "build raised (isolated; sweep continues)"))
            return norm_cell({**base, "status": _st, "reason": _rs, **_err_fields(err)})
        return norm_cell({**base, "status": "error",
                          "reason": "peer rank build failed (skipped in lockstep; this rank built ok)"})
    # 4. optional correctness gate (all ranks agree via all_reduce)
    correct = None
    if do_gate and target.output is not None and target.ref is not None:
        try:
            ok = bool(_gate_output(target.output(handle), target.ref(ctx)))
        except Exception:
            ok = False
        agree = _consensus_ok(ctx, ok)  # all_reduce the pass flag -> symmetric
        correct = agree
        if not agree:
            if target.teardown is not None:
                try:
                    target.teardown(handle)
                except Exception:
                    pass
            return norm_cell({**base, "status": "disqualified_wrong", "correct": False,
                              "reason": "correctness gate failed (or a peer's did)"})
    # 5. timing (entered by ALL ranks or NONE)
    time_ms = None
    raw_ms = None
    err2 = None
    try:
        # STOPGAP fail-quick: bound the TIMED hot-loop so a 2-node cluster-drain in-kernel deadlock (e.g.
        # the documented multislot 2-slot straddle L=1 hang) becomes a fast consensus'd skip, not a
        # 48-min freeze. A CUDA/NCCL sync is interruptible by SIGALRM at a Python boundary; a fully wedged
        # GPU is bounded by the raised dist-store timeout (raise_store_timeout) as the last backstop.
        with phase_deadline(_RUN_BUDGET, "run", N=ctx.N, cfg=dict(ctx.cfg)):
            time_ms, raw_ms = _timeit(target, handle, ctx, rounds=rounds, warmup=warmup)
    except Exception as e:
        err2 = e
    # 6. teardown (always)
    if target.teardown is not None:
        try:
            target.teardown(handle)
        except Exception:
            pass
    # a timing raise is ALSO consensus'd so all ranks agree on the outcome
    time_ok = _consensus_ok(ctx, err2 is None)
    if not time_ok:
        _run_to = err2 is not None and type(err2).__name__ == "PhaseTimeout"
        return norm_cell({**base, "status": ("timeout" if _run_to else "error"), "correct": correct,
                          "reason": ("run TIMEOUT (fail-quick stopgap; likely a 2-node drain deadlock)"
                                     if _run_to else "run/timing raised (isolated)"),
                          **(_err_fields(err2) if err2 is not None else {})})
    # 7. OPTIONAL comm-BW metric (None-guarded -> back/fused targets unaffected). comm_bytes(ctx) is the
    # per-rank off-diagonal A2A byte volume V_a2a; time_ms is the all_reduce_MAX (slowest-PE) round -> the
    # emitted GB/s is the per-GPU achieved injection BW (comparable to beta_c ~332/359 NVLink, beta_ib ~47-49).
    return norm_cell({**base, "status": "ok", "correct": correct, "time_ms": time_ms, "raw_ms": raw_ms,
                      **_comm_bw_fields(target, ctx, time_ms), **_cell_meta_fields(target, ctx)})


def _cell_meta_fields(target, ctx):
    """Merge target.cell_meta(ctx) (a dict of extra cell fields, e.g. gemm_cfg) into the ok cell. None-guarded
    -> targets without the hook (all back/comm targets) are unaffected."""
    fn = getattr(target, "cell_meta", None)
    if fn is None:
        return {}
    try:
        d = fn(ctx)
    except Exception:
        return {}
    return dict(d) if isinstance(d, dict) else {}


def _node_sz(ctx):
    """Intra-NVLink-domain GPU count for the IB/NVLink comm split. Convention: the (cp0,cp1) mesh is
    (nodes, within-node) (rp._cp_axis_sizes), so node_sz = cp1 (cp1==1 => cross-node 1-D => node_sz=1,
    all off-diagonal crosses IB). Env CPO_HARNESS_NODE_SZ overrides for a single-node 1-D cp run."""
    cp = int(ctx.cp0) * int(ctx.cp1)
    env = os.environ.get("CPO_HARNESS_NODE_SZ")
    if env:
        try:
            return max(1, min(int(env), cp))
        except ValueError:
            pass
    ns = int(ctx.cp1) if int(ctx.cp1) > 1 else 1
    return max(1, min(ns, cp))


def _comm_bw_fields(target, ctx, time_ms):
    """{comm_bytes, node_sz, gbps_total, gbps_ib, gbps_nvl} for a comm target, else {} (None-guarded). The
    IB/NVLink split partitions V_a2a's (cp-1) off-diagonal peers: (cp-node_sz) cross IB, (node_sz-1) on
    NVLink (design-doc layernorm_dual_gated_gemm_a2a_design.md §0.4c V_ib/V_nvl)."""
    cb = getattr(target, "comm_bytes", None)
    if cb is None or time_ms is None or time_ms <= 0:
        return {}
    try:
        nbytes = int(cb(ctx))
    except Exception:
        return {}
    if nbytes <= 0:
        return {}
    cp = int(ctx.cp0) * int(ctx.cp1)
    node_sz = _node_sz(ctx)
    gbps_total = nbytes / (time_ms * 1e6)   # bytes / (ms * 1e6) = GB/s
    out = {"comm_bytes": nbytes, "node_sz": node_sz, "gbps_total": gbps_total}
    if cp > 1:   # split the off-diagonal by locality (fractions of V_a2a): IB=(cp-node_sz), NVLink=(node_sz-1)
        out["gbps_ib"] = gbps_total * (cp - node_sz) / (cp - 1)
        out["gbps_nvl"] = gbps_total * (node_sz - 1) / (cp - 1)
    return out


def run_cell(target, ctx, *, do_gate, rounds, warmup):
    """A CELL = one target at one shape, aggregating its autotune configs. Returns {target, configs[], winner,
    status}. cfgs from target.configs(ctx) (deterministic on all ranks) or a single {} config."""
    try:
        cfgs = target.configs(ctx) if target.configs is not None else [{}]
    except Exception:
        cfgs = [{}]
    if not cfgs:
        cfgs = [{}]
    if _MAX_CONFIGS and len(cfgs) > _MAX_CONFIGS:   # smoke/quick mode: deterministic prefix -> same on all ranks
        cfgs = cfgs[:_MAX_CONFIGS]
    subs = []
    for cfg in cfgs:
        # LOCKSTEP ENTRY (mirrors back_a2a_store_bench:848 + the :715 post-free barrier): an UNCONDITIONAL
        # all-rank barrier at every config boundary. It drains the PRIOR config's symmetric-mem free (teardown)
        # on every rank before this config allocs -> keeps nvshmem symmetric alloc/free lockstep across cells.
        # Without it, per-cell alloc(build)+free(teardown) races across ranks -> heap desync -> illegal access
        # at a later barrier (the coalesce->differential smoke crash). Unconditional -> collective-symmetric.
        _barrier(ctx)
        subs.append(run_one_config(target, ctx.with_cfg(cfg), do_gate=do_gate, rounds=rounds, warmup=warmup))
    valid = [s for s in subs if s.get("status") == "ok" and s.get("time_ms") is not None]
    winner = min(valid, key=lambda s: s["time_ms"]) if valid else None
    if valid:
        status, reason, extra = "ok", "", {}
    else:
        errs = [s for s in subs if s.get("status") in ("error", "oom")]
        skips = [s for s in subs if s.get("status") == "skip_shape"]
        if errs:
            status = "oom" if all(s.get("status") == "oom" for s in errs) else "error"
            reason = f"all {len(subs)} config(s) failed; first: {(errs[0].get('error_msg') or '')[:200]}"
            extra = {"error_class": errs[0].get("error_class"), "error_msg": errs[0].get("error_msg")}
        elif skips and len(skips) == len(subs):
            status, reason, extra = "skip_shape", skips[0].get("reason", ""), {}
        else:
            status, reason, extra = "error", f"no valid config among {len(subs)}", {}
    return norm_cell({"target": target.name, "role": target.role, "N": ctx.N, "cp0": ctx.cp0, "cp1": ctx.cp1,
                      "Dloc": ctx.Dloc, "configs": subs, "winner": winner, "n_valid": len(valid),
                      "status": status, "reason": reason, **extra})


def run_matrix(targets, shapes, ctx_base, *, rounds=5, warmup=1, do_gate=False, first_n=None, out_dir=None,
               rank0=True, log=None):
    """§2: iterate shape × target -> cells. Emits one JSON per shape (cells[]) + returns the full cell list +
    an (shape × target) -> status matrix. Isolation at every level: a bad target -> error cell + continue; a
    bad shape -> the whole shape's cells error + continue; only OOM/timeout is fatal (and OOM is a cell)."""
    _log = log or (print if rank0 else (lambda *a, **k: None))
    all_cells = []
    matrix = {}  # (N, target) -> status
    for N in shapes:
        _dg = bool(do_gate and (first_n is None or N == first_n))  # gate only at the anchor N (perf-only)
        cells = []
        try:
            for t in targets:
                ctx = _with_N(ctx_base, N)
                try:
                    cell = run_cell(t, ctx, do_gate=_dg, rounds=rounds, warmup=warmup)
                except Exception as e:  # whole-cell blowup -> error cell, next target continues
                    cell = norm_cell({"target": t.name, "role": t.role, "N": N, "status": "error",
                                      "reason": "cell-level exception (isolated)", "winner": None,
                                      "configs": [], **_err_fields(e)})
                cells.append(cell)
                all_cells.append(cell)
                matrix[(N, t.name)] = cell["status"]
                _log(f"  N={N:>7} {t.name:>18} [{t.role}] -> {cell['status']} "
                     f"winner={_wsummary(cell.get('winner'))} reason={cell.get('reason','')[:50]}")
                if rank0 and out_dir:   # INCREMENTAL flush: a later hard crash (CUDA abort -> process death,
                    _flush_json(out_dir, N, ctx_base, cells)   # uncatchable) keeps every cell finished so far
        except Exception as e:   # shape-level blowup -> continue to next shape
            _log(f"[N-ERROR] N={N}: {type(e).__name__}: {repr(e)[:160]}")
            all_cells.append(norm_cell({"target": "*", "N": N, "status": "error",
                                        "reason": "shape-loop exception (isolated)", **_err_fields(e)}))
        if rank0 and out_dir:
            _flush_json(out_dir, N, ctx_base, cells)   # final authoritative write for this shape
    if rank0:
        _print_matrix(all_cells, targets, shapes, _log)
    return {"cells": all_cells, "matrix": matrix}


def _flush_json(out_dir, N, ctx_base, cells):
    """Write the shape's cells to bench_N{N}.json (rank0). Called after EVERY cell (incremental crash-safety)
    and once more at shape end. Atomic-ish: write a tmp then replace, so a crash mid-write can't corrupt it.

    **PER CELL, and a hard crash is only half the reason.** The other half is the one that bites a
    new `BenchTarget` author writing their own accumulator: an end-of-run-only write UNDER-REPORTS
    SILENTLY EXACTLY WHEN TRUNCATION IS WORKING AS DESIGNED. A bounded run that hits its own
    `timeout` is not a crash -- it is the intended behaviour, since CLAUDE.md caps a bound at one
    hour and per-cell isolation exists so a truncated sweep resumes. A writer that only flushes at
    the end therefore loses everything such a run produced, and reads correct on every run that
    never needed the feature.

    Measured on a bespoke driver that got this wrong (`w8plan/diag/backpins.sh`, 2026-08-20): its
    ws=8 harvest was killed at 30 of 32 cells by its own bound, and the 232 harvest lines it had
    already collected never reached the aggregate -- 536 on file against 768 in the per-cell logs.
    A merge fed from that aggregate would have written 536 lines' worth of pins and REPORTED
    SUCCESS. Nothing was lost only because the per-cell artifacts are the record, which is the
    property this function provides and a hand-rolled accumulator does not."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"bench_N{N}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"N": N, "cp0": ctx_base.cp0, "cp1": ctx_base.cp1, "cells": cells}, f, indent=2)
    os.replace(tmp, path)


def _with_N(ctx, N):
    return type(ctx)(dm=ctx.dm, pm=ctx.pm, da=ctx.da, N=N, cp0=ctx.cp0, cp1=ctx.cp1, Dloc=ctx.Dloc, B=ctx.B,
                     rd=ctx.rd, device=ctx.device, rank=ctx.rank, cfg=dict(ctx.cfg))


def _wsummary(w):
    if not isinstance(w, dict):
        return "-"
    return f"{w.get('time_ms')}ms cfg={w.get('cfg')}"


def _print_matrix(all_cells, targets, shapes, log):
    glyph = {"ok": "ok", "error": "ER", "oom": "OM", "skip_shape": "sk", "disqualified_wrong": "XX",
             "timeout": "TO"}
    by, counts = {}, {}
    for c in all_cells:
        st = str(c.get("status"))
        counts[st] = counts.get(st, 0) + 1
        by[(c.get("N"), c.get("target"))] = st
    names = [t.name for t in targets]
    log("\n============ BENCH MATRIX (shape × target → status) ============")
    log(f"{'N':>8} | " + " ".join(f"{n[:10]:>10}" for n in names))
    for N in shapes:
        log(f"{N:>8} | " + " ".join(f"{glyph.get(by.get((N, n)), '--'):>10}" for n in names))
    log("counts: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    log("================================================================\n")
