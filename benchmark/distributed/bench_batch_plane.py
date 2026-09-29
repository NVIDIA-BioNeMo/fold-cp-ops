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

"""Paired A/B of the INCOMING fast front store against its plain control, at B=1 and B=2.

Purpose
    Answer two questions the correctness gate cannot: does carrying a batch plane over the IB
    drain cost anything at ``B == 1`` (neutrality, measured ACROSS TREES), and does the fast
    store's runtime scale with the batch extent the way its plain control does (measured WITHIN
    one tree, in one process). Both answers are RATIOS, which is what makes this script valid on
    a venue whose absolute numbers are not trustworthy: a systematic slowdown that hits both arms
    cancels.

Semantics
    Builds TWO engines on the same mesh, weights and shard -- one with the requested fast
    incoming-store flag (``composite_k`` or ``route2_ni``) and one with neither, which is the
    PLAIN store -- and times four arms interleaved per round through
    :func:`fold_cp_ops._internal.bench_timing.benchmark_paired`:

    ``fast_b1`` / ``fast_b2`` / ``plain_b1`` / ``plain_b2``.

    WHAT THIS HARNESS HAS MEASURED, so a reader need not re-run it to know the answer. Both
    variants are NEUTRAL at ``B == 1``, each over MATCHED-ARM cells (``--batches 1``, so both trees
    time exactly two arms):

    * ``composite_k`` -- **+-1.03%** over 6 cells, venue E DGX1/DGX2: cp16 cross-node IB
      (D256 x N4096/6144/8192, D512 x N4096/6144) and cp8 NVLink (D256 N6144).
    * ``route2_ni``   -- **+-0.54%** over 5 cells cross-node IB (cp4x4: D256 x N4096/6144/8192,
      D512 x N4096/6144) and **+0.13%..+0.25%** over 3 cells on NVLink (cp2x4: D256 x N4096/6144,
      D512 x N4096). Venue B, multi-rail verified.

    AND ONE THING THIS HARNESS FOUND THAT IS NOT ABOUT NEUTRALITY, recorded because the next reader
    will otherwise re-derive it. ``fp_mine``/``fp_base`` -- the fast store over its plain control --
    flips SIGN with the TRANSPORT, not with the venue:

        cp2x4  NVLink     0.71-0.79   fast is FASTER than plain
        cp4x4  cross-node 1.09-1.61   fast is SLOWER than plain

    Same cluster, same container, same harness, same two trees, one launch apart -- so venue,
    build and measurement method are all held fixed and only the transport moves. It is present
    EQUALLY in both trees (``fp_mine ~= fp_base`` to within 0.5%), so it is NOT the batch plane and
    cannot bear on neutrality. It is an open question about the route2_ni store over IB.

    AND THE REASON ``--batches 1`` IS NOT A CONVENIENCE. The same ``route2_ni`` D512 N4096 cell read
    **0.9559** -- an apparent 4.4% regression -- when measured with ``--batches 1,2``, because the
    pre-widening tree REFUSES ``fast_b2`` cross-node and therefore timed 3 arms against this tree's
    4. Re-measured with arm counts matched, it reads **0.9990**. The 4.4% was the duty-cycle
    confound of ``4a8fba3``, not the code. A cross-tree B=1 comparison is only honest when both
    trees run the same arm set; at cp=16 and cp4x4 they can do that only at ``--batches 1``.

    A bare :class:`TriMulAutotuned` does NOT apply ``incoming_store_variant`` itself, so the
    plain arm is obtained simply by passing neither flag, and the fast arm by passing exactly one.
    Both engines are built with ``dynamic=True`` and the anchor ``B = 1`` so a single compile
    serves both batch extents -- the same construction the correctness gate uses.

    ``benchmark_paired`` times each arm in its own barrier-bracketed CUDA-event window, one
    window per arm per round, so a clock or thermal excursion inside a round scales every arm
    together. This script then reduces each round's per-arm ms across ranks with
    ``all_reduce(MAX)`` -- making every reported number a SLOWEST-PE quantity, i.e. a property of
    the job rather than of whichever rank finished first -- and only then takes medians and
    ratios. That MAX is why ``benchmark_paired``'s own ``median_ratio`` is ignored: it is
    per-rank, and a collective workload's cost is the slowest rank's.

    Autotune is explicitly OFF (``autotune_config=False``) rather than left to
    ``CPO_DIST_AUTOTUNE``. The subject is the STORE, and a tuner free to pick a different config
    in the two trees would put a config difference into a ratio that is supposed to isolate the
    store.

    Arms that raise are dropped rank-uniformly (the decision is ``all_reduce``-ed) and recorded
    with the exception text, never silently omitted: on the pre-batch-plane tree the cross-node
    fast store at ``B > 1`` is expected to refuse, and "refused" must be distinguishable from
    "was not run".

Input requirements
    ``--mesh``   one of the keys of :data:`MESHES`; its rank product MUST equal ``WORLD_SIZE`` or
                 the script exits 3 with a message. ``cp8``/``cp2x4`` need 8 ranks, ``cp16``/
                 ``cp4x4`` need 16.
    ``--D``      full feature width; must be divisible by the mesh's rank product (the per-peer
                 width is ``D // cp``) or the build raises deep inside the engine.
    ``--N``      token extent; must be divisible by BOTH mesh axes, and the resulting per-peer
                 extent must clear the 16-byte floor (bf16 -> a multiple of 8). All declared
                 values (4096, 8192) satisfy this on every mesh here.
    ``--rounds`` timed rounds per arm; ``--warmup`` untimed full windows; ``--iters`` back-to-back
                 launches per window. Defaults are sized for the tens-of-ms cells this measures,
                 where the ~0.3 ms per-call host dispatch is already negligible at ``iters=5``.
    Launch      under torchrun/srun with ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` set; one rank per
                 GPU. ``PYTHONPATH`` must contain the repo root, because the weight builder is
                 ``tests.distributed.correctness_harness.make_weights`` -- the same builder the
                 correctness gate uses, so a perf cell and a correctness cell are the same tensor.

Returns / Raises
    Writes one JSON object per cell to ``--out`` (rank 0 only) and prints a ``[CELL]`` line per
    cell plus a final ``[DENOM]`` line carrying ``requested=/ran=/ok=`` and an ``INCOMPLETE``
    marker when they disagree -- a truncated sweep must never read as a clean one.
    Exits 0 when every requested cell produced a result, 3 on a mesh/world mismatch, 4 when at
    least one cell failed, 5 on a setup failure before any cell ran.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from collections import OrderedDict

#: label -> (cp0, cp1). ``cp1 == 1`` is the composite_k side of `incoming_store_variant`; any
#: ``cp1 > 1`` is the route2_ni side. Kept as a literal table rather than parsed from the label so
#: a typo is a KeyError here instead of a silently different mesh.
MESHES = {
    "cp8": (8, 1),
    "cp16": (16, 1),
    "cp2x4": (2, 4),
    "cp4x4": (4, 4),
    "cp2x2": (2, 2),
    "cp2x8": (2, 8),
    "cp4": (4, 1),
    "cp2": (2, 1),
}


def _mesh_spec(label):
    """``grid_group_sizes`` for ``create_grid_group`` from a :data:`MESHES` label.

    A flat mesh is passed as a bare int and a factorised one as a tuple, because
    ``create_grid_group`` builds the ``<name>_axis_<i>`` subgroups only for the tuple form -- and
    without those subgroups ``cp_axis_sizes`` reads ``(cp,)``, ``cp1`` collapses to 1 and every
    2-D mesh would select ``composite_k``.

    Args:
        label: a key of :data:`MESHES`; anything else raises ``KeyError``.

    Returns:
        ``OrderedDict`` with the single key ``"cp"``.
    """
    a, b = MESHES[label]
    return OrderedDict([("cp", a if b == 1 else (a, b))])


def _variant_for(label):
    """The fast incoming-store flag this mesh admits: ``composite_k`` when ``cp1 == 1`` else
    ``route2_ni``.

    Mirrors ``incoming_store_variant(direction="incoming", cp1=...)`` without importing it, so the
    script is identical on a tree that predates that helper. ``composite_k`` additionally REQUIRES
    ``cp1 == 1`` inside the engine (it raises otherwise), which is the same condition, so the two
    can not drift apart.

    Args:
        label: a key of :data:`MESHES`.

    Returns:
        ``"composite_k"`` or ``"route2_ni"``.
    """
    return "composite_k" if MESHES[label][1] == 1 else "route2_ni"


def _rank0_print(rank, *a, **kw):
    """Print only on rank 0, always flushed. A per-rank print at 16 ranks interleaves into an
    unreadable log and, worse, makes a per-rank divergence look like ordinary repetition."""
    if rank == 0:
        print(*a, **kw, flush=True)


def _max_across_ranks(vals, device):
    """Element-wise ``all_reduce(MAX)`` of a per-round ms vector, so each round reports the
    SLOWEST PE.

    The vectors are aligned across ranks because ``benchmark_paired`` barrier-brackets every
    window, so round *i* on one rank is round *i* on every other.

    Args:
        vals: this rank's per-round ms, one float per timed round. Every rank MUST pass the same
            length -- an arm dropped on one rank only would deadlock here, which is why the drop
            decision is all-reduced before this is reached.
        device: the CUDA device to stage the reduction on.

    Returns:
        A new list of the same length holding the per-round maxima.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return list(vals)
    t = torch.tensor(list(vals), dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return [float(v) for v in t.tolist()]


def _any_rank(flag, device):
    """True when ``flag`` is true on ANY rank -- the rank-uniform form of "this arm refused".

    A per-rank decision to drop an arm is a collective divergence: the dropping rank skips a
    window every peer still enters. This turns the decision into one number every rank agrees on.

    Args:
        flag: this rank's local boolean.
        device: CUDA device for the reduction.

    Returns:
        The OR of ``flag`` over the world.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return bool(flag)
    t = torch.tensor([1.0 if flag else 0.0], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0.5)


def _build_engines(pm, mesh, placements, N, D, weights, dt, variant, which="both"):
    """Build the fast-store engine and its plain control on the same mesh and weights.

    Both are ``dynamic=True`` with the anchor ``B = 1``, so one compile serves B=1 and B=2 and the
    batch extent stays a runtime input -- constructing at ``B = 2`` instead would, on the
    pre-batch-plane tree, trip the cross-node demotion at construction and silently give the fast
    arm the plain kernel.

    Args:
        pm: this job's ``PeMap``.
        mesh / placements: the DeviceMesh and per-dim ``Shard`` placements the PeMap was built from.
        N, D: full token and feature extents (not per-peer).
        weights: the ``make_weights`` dict.
        dt: element dtype (``torch.bfloat16`` here).
        variant: ``"composite_k"`` or ``"route2_ni"`` -- the flag set on the fast engine.
        which: ``"both"`` (default), ``"fast"`` or ``"plain"``. ONE engine halves the resident
            symmetric footprint, which is the difference between measuring and OOM-ing at the top
            of the ladder: at cp=16 / N=8192 / D=256 the pair fits at B=1 and the B=2 recv then
            cannot be allocated, so the batch-scaling arms are lost to the harness rather than to
            the kernel. ``"both"`` is required for the fast/plain ratio (it must be within one
            process to cancel this venue's launch-level excursions); the single forms are for the
            B2/B1 ratio, which never crosses engines.

    Returns:
        ``(fast, plain, meta)`` where the engine not requested is ``None`` and ``meta`` records
        what each engine actually BUILT, read back off the objects rather than assumed, plus the
        resolved ``hybrid_ib``.

    Raises:
        Whatever the engine raises: a ``ValueError`` for an illegal flag/mesh combination, or
        ``torch.cuda.OutOfMemoryError`` from the symmetric allocation.
    """
    from fold_cp_ops.distributed.workflows.trimul_autotuned import TriMulAutotuned

    common = dict(
        device_mesh=mesh,
        placements=placements,
        dynamic=True,
        autotune_config=False,
    )
    fast = plain = None
    if which in ("both", "fast"):
        fast = TriMulAutotuned(pm, 1, N, D, weights, dt, **{variant: True}, **common)
    if which in ("both", "plain"):
        plain = TriMulAutotuned(pm, 1, N, D, weights, dt, **common)

    def _built(e):
        if e is None:
            return None
        return "composite_k" if e.composite_k else ("route2_ni" if e.route2_ni else "plain")

    meta = {
        "engines": which,
        "fast_built": _built(fast),
        "plain_built": _built(plain),
        "hybrid_ib": bool((fast or plain).hybrid_ib),
    }
    return fast, plain, meta


def _run_cell(args, dm, pm, mesh, placements, N, D, variant, device):
    """Measure one (N, D) cell: build, warm, time four interleaved arms, reduce, summarise.

    Semantics
        The pre-warm is EXPLICIT and outside the timed region so a first-call JIT compile is
        attributed to compile rather than to the arm, and so an arm that refuses does so where it
        can be caught and reported. ``benchmark_paired``'s own warmup rounds then follow.

    Args:
        args: the parsed CLI namespace (``rounds`` / ``warmup`` / ``iters`` / ``seed``).
        dm: the live ``DistributedManager``.
        pm / mesh / placements: as in :func:`_build_engines`.
        N, D: full extents for this cell.
        variant: the fast flag to set.
        device: this rank's CUDA device.

    Returns:
        A JSON-ready dict with per-arm slowest-PE medians, the four derived ratios, and the
        per-arm status. Never raises for an arm-level refusal; a build-level failure propagates
        to the caller, which records the cell as failed.
    """
    import torch

    from fold_cp_ops._internal.bench_timing import benchmark_paired
    from tests.distributed.correctness_harness import make_weights

    cp0, cp1 = pm_axes(pm)
    ni, nj = N // cp0, N // cp1
    dt = torch.bfloat16
    rank = int(dm.rank)

    want_b2 = "2" in str(args.batches).split(",")
    weights = make_weights(D, seed=args.seed, device=device)
    g = torch.Generator(device="cpu").manual_seed(args.seed + 5)
    # Built at B=2 and sliced, so the B=1 arm reads the SAME values as plane 0 of the B=2 arm.
    # `contiguous()` on an already-contiguous outer slice returns self, so `x1` costs no memory.
    x2 = torch.randn(2 if want_b2 else 1, ni, nj, D, generator=g, dtype=torch.float32)
    x2 = x2.to(device).to(dt)
    x1 = x2[:1].contiguous()

    t0 = time.time()
    fast, plain, meta = _build_engines(
        pm, mesh, placements, N, D, weights, dt, variant, which=args.engines
    )
    build_s = time.time() - t0

    result = {
        "N": N,
        "D": D,
        "cp0": cp0,
        "cp1": cp1,
        "cp": cp0 * cp1,
        "variant": variant,
        "build_s": round(build_s, 2),
        **meta,
        "batches": str(args.batches),
        "rounds": args.rounds,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    try:
        # Arm ORDER is fast_b1, fast_b2, plain_b1, plain_b2 and is fixed, because every arm's window
        # is measured in the thermal wake of the one before it.
        cand = []
        for tag, eng in (("fast", fast), ("plain", plain)):
            if eng is None:
                continue
            cand.append((f"{tag}_b1", (eng, x1)))
            if want_b2:
                cand.append((f"{tag}_b2", (eng, x2)))
        candidates = OrderedDict(cand)
        targets, status = OrderedDict(), {}
        t0 = time.time()
        for label, (eng, x) in candidates.items():
            local_err = ""
            try:
                eng.forward(x, "incoming")
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001 -- a refusal is a RESULT here, not a crash
                local_err = f"{type(e).__name__}: {e}"[:300]
            failed = _any_rank(bool(local_err), device)
            if failed:
                status[label] = local_err or "refused on another rank"
                continue
            status[label] = "ok"
            targets[label] = lambda e=eng, xx=x: e.forward(xx, "incoming")
        result["warm_s"] = round(time.time() - t0, 2)
        result["arm_status"] = status
        if not targets:
            result["error"] = "every arm refused"
            return result

        paired = benchmark_paired(
            targets,
            rounds=args.rounds,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        med = {}
        for label in targets:
            entry = paired.targets.get(label, {})
            raw = entry.get("raw_ms")
            if not raw:
                status[label] = entry.get("status", "no timed rounds")
                continue
            mx = _max_across_ranks(raw, device)
            med[label] = statistics.median(mx)
            result[f"raw_max_{label}"] = [round(v, 5) for v in mx]
        result["median_ms"] = {k: round(v, 5) for k, v in med.items()}

        def _ratio(a, b):
            return round(med[a] / med[b], 4) if a in med and b in med and med[b] > 0 else None

        result["ratios"] = {
            "fast_b2_over_b1": _ratio("fast_b2", "fast_b1"),
            "plain_b2_over_b1": _ratio("plain_b2", "plain_b1"),
            "fast_over_plain_b1": _ratio("fast_b1", "plain_b1"),
            "fast_over_plain_b2": _ratio("fast_b2", "plain_b2"),
        }
    finally:
        # `fast`/`plain` are bound above the `try`, so a build failure never reaches here and both
        # names are always live when it does; either may be None under `--engines fast|plain`.
        for e in (fast, plain):
            if e is None:
                continue
            try:
                e.free()
            except Exception:  # noqa: BLE001 -- teardown must not mask the measurement
                _rank0_print(rank, "[warn] engine.free() raised:\n" + traceback.format_exc())
        del x1, x2
        torch.cuda.empty_cache()
    return result


def pm_axes(pm):
    """``(cp0, cp1)`` from a ``PeMap``, with ``cp1 == 1`` for a flat mesh.

    Args:
        pm: a built ``PeMap``; ``cp_axis_sizes`` must be populated.

    Returns:
        Two ints whose product is the cp rank count.
    """
    a = int(pm.cp_axis_sizes[0])
    b = int(pm.cp_axis_sizes[1]) if len(pm.cp_axis_sizes) > 1 else 1
    return a, b


def summarize(directory):
    """Reduce a directory of per-launch JSON files to the two claims, and print them.

    Purpose
        Each launch writes one JSON per tree per process round. The claims are ratios ACROSS
        those files, so the reduction has to happen somewhere; putting it in the measuring
        script keeps one artifact per subject and removes any chance of a second, drifting
        definition of "the ratio".

    Semantics
        Files are grouped by ``(mesh, D, N, tree)`` where the tree is read from the ``commit``
        field the driver stamped -- NOT from the filename, which a rename would silently
        falsify. Within a group the per-round values are reduced with a MEDIAN, which is what
        makes a whole-launch excursion (measured on this venue: one round ran every arm ~1.5x
        slow) an outlier rather than a bias.

        Two neutrality numbers are reported and they are not redundant:

        * ``neutral_within`` = ``(fast_b1/plain_b1)_mine / (fast_b1/plain_b1)_base``. Both
          halves are WITHIN-process, per-round interleaved ratios, so every process-level
          offset -- clock, thermal, allocator, heap address -- cancels twice. The plain store
          is untouched by the change, which is what makes it a fixed reference in both trees.
        * ``neutral_cross`` = ``median(fast_b1)_mine / median(fast_b1)_base``, the direct
          cross-process comparison. It carries the process-level noise the first one removes,
          and is reported because agreement between the two is the evidence that neither is an
          artefact of its own reduction.

    Input requirements
        ``directory`` holds ``*.json`` files written by this script's measuring mode; anything
        else is skipped. A cell missing from one tree is reported with ``-`` rather than
        dropped, so a hole is visible.

    Returns:
        0 always -- this is a reporting mode, and a missing cell is reported, not raised.
    """
    import glob

    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path) as fh:
                rows.extend(json.load(fh))
        except (OSError, ValueError):
            print(f"[skip] unreadable {path}")
    by = {}
    for r in rows:
        if not r.get("median_ms"):
            continue
        tree = "mine" if str(r.get("commit", "")).startswith("fb846de") else "base"
        # `engines` is part of the key, not a footnote: a one-engine launch has HALF the resident
        # symmetric footprint, so its arms run in a different memory regime. Pooling the two would
        # average a paired ratio with an unpaired one and print a single number for two things.
        key = (r["mesh"], r["D"], r["N"], r.get("engines", "both"), r.get("batches", "1,2"))
        by.setdefault(key, {}).setdefault(tree, []).append(r)

    def _med(rs, key):
        vals = [x["median_ms"][key] for x in rs if key in x["median_ms"]]
        return statistics.median(vals) if vals else None

    def _med_ratio(rs, a, b):
        """Median over ALL rounds of the PER-ROUND ratio ``a/b``, pooled over process rounds.

        Per-round rather than ratio-of-medians, and that is a correction with a measurement behind
        it: each launch ramps monotonically across its timed rounds (cp16/D512/N4096, one launch:
        ``fast_b1`` 290 -> 347 ms over 9 rounds), so a median lands mid-ramp and its position
        depends on how much work the round carries. Two arms timed in the SAME round sit at the
        same point on that ramp, so their ratio is what the ramp cancels out of.
        """
        vals = []
        for x in rs:
            ra, rb = x.get("raw_max_" + a), x.get("raw_max_" + b)
            if ra and rb and len(ra) == len(rb):
                vals += [p / q for p, q in zip(ra, rb) if q > 0]
            elif a in x.get("median_ms", {}) and x.get("median_ms", {}).get(b, 0) > 0:
                vals.append(x["median_ms"][a] / x["median_ms"][b])
        return statistics.median(vals) if vals else None

    hdr = (
        f"{'mesh':7} {'variant':11} {'eng':5} {'bat':4} {'D':>4} {'N':>6} {'n_m':>3} {'n_b':>3} "
        f"{'neutral_within':>14} {'neutral_cross':>13} {'fastB2/B1':>10} {'plainB2/B1':>11} "
        f"{'fp_mine':>8} {'fp_base':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for mesh, D, N, eng, bat in sorted(by):
        g = by[(mesh, D, N, eng, bat)]
        m, b = g.get("mine", []), g.get("base", [])
        variant = (m or b)[0].get("variant", "?")
        fpm, fpb = _med_ratio(m, "fast_b1", "plain_b1"), _med_ratio(b, "fast_b1", "plain_b1")
        f1m, f1b = _med(m, "fast_b1"), _med(b, "fast_b1")
        nw = f"{fpm / fpb:.4f}" if fpm and fpb else "-"
        nc = f"{f1m / f1b:.4f}" if f1m and f1b else "-"
        fb = _med_ratio(m, "fast_b2", "fast_b1")
        pb = _med_ratio(m, "plain_b2", "plain_b1")
        print(
            f"{mesh:7} {variant:11} {eng:5} {bat:4} {D:>4} {N:>6} {len(m):>3} {len(b):>3} "
            f"{nw:>14} {nc:>13} {(f'{fb:.4f}' if fb else '-'):>10} "
            f"{(f'{pb:.4f}' if pb else '-'):>11} {(f'{fpm:.4f}' if fpm else '-'):>8} "
            f"{(f'{fpb:.4f}' if fpb else '-'):>8}"
        )
    holes = [k for k, g in by.items() if not g.get("mine") or not g.get("base")]
    errs = [(r.get("mesh"), r.get("D"), r.get("N"), r.get("error")) for r in rows if r.get("error")]
    print(f"\ncells={len(by)}  one-sided={len(holes)}  errored_rows={len(errs)}")
    for e in errs:
        print(f"  [ERR] {e}")
    return 0


def main(argv=None):
    """Parse the CLI, bring up the world and the mesh, and walk the requested cells.

    Args:
        argv: argument vector; ``None`` uses ``sys.argv[1:]``.

    Returns:
        A process exit code: 0 all-ok, 3 mesh/world mismatch, 4 at least one cell failed,
        5 setup failed before any cell ran.
    """
    if argv is None and len(sys.argv) > 2 and sys.argv[1] == "--summarize":
        return summarize(sys.argv[2])
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", required=True, choices=sorted(MESHES))
    p.add_argument(
        "--cells", required=True, help="comma-separated D:N pairs, e.g. '256:4096,512:8192'"
    )
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument(
        "--batches",
        default="1,2",
        help="'1,2' times B=1 AND B=2 (the batch-scaling claim); '1' times B=1 only. "
        "MEASURED and load-bearing: every launch ramps thermally across its timed "
        "rounds, and a 4-arm launch ramps faster than a 3-arm one -- so a "
        "cross-tree B=1 comparison is only honest when BOTH trees run the same "
        "arm set, which at cp=16 they can do only at '1' (base refuses fast_b2)",
    )
    p.add_argument(
        "--engines",
        default="both",
        choices=("both", "fast", "plain"),
        help="'both' pairs fast against plain in one process (required for the "
        "fast/plain ratio); 'fast'/'plain' build ONE engine, halving the resident "
        "symmetric footprint so the B=2 arms survive the top of the ladder",
    )
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--out", default="")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    cells = []
    for tok in args.cells.split(","):
        d, n = tok.split(":")
        cells.append((int(d), int(n)))

    import torch

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    cp0, cp1 = MESHES[args.mesh]
    if cp0 * cp1 != world:
        _rank0_print(rank, f"[REFUSED] mesh {args.mesh} needs {cp0 * cp1} ranks, world={world}")
        return 3

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("CPO_DISTRIBUTED_INIT_METHOD", "ENV")

    from fold_cp_ops.distributed.distributed_manager import DistributedManager
    from fold_cp_ops.distributed.pe_map import PeMap
    from torch.distributed.tensor import Shard

    t0 = time.time()
    DistributedManager.initialize(None, device_type="cuda", backend="nccl")
    if not torch.distributed.is_initialized():
        _rank0_print(rank, "[REFUSED] no process group after initialize()")
        return 5
    DistributedManager.init_nvshmem()
    DistributedManager.reset_grid_groups()
    DistributedManager.create_grid_group(_mesh_spec(args.mesh))
    dm = DistributedManager()
    sub = getattr(dm, "device_mesh_subgroups", None)
    mesh = sub if sub is not None else dm.device_mesh
    placements = [Shard(i + 1) for i in range(mesh.ndim)]
    pm = PeMap.from_mesh_placements(mesh, placements, distributed_manager=dm)
    device = dm.device
    init_s = time.time() - t0

    variant = _variant_for(args.mesh)
    got0, got1 = pm_axes(pm)
    commit = ""
    for base in (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),):
        try:
            with open(os.path.join(base, "STAGED_FROM_COMMIT.txt")) as fh:
                commit = fh.read().strip()[:12]
        except OSError:
            pass
    _rank0_print(
        rank,
        f"[SETUP] tag={args.tag} commit={commit} mesh={args.mesh} declared=({cp0},{cp1}) "
        f"pe_map=({got0},{got1}) variant={variant} world={world} init_s={init_s:.1f}",
    )
    if (got0, got1) != (cp0, cp1):
        _rank0_print(rank, f"[REFUSED] PeMap reports ({got0},{got1}), expected ({cp0},{cp1})")
        return 5

    out_rows, n_ok = [], 0
    for D, N in cells:
        row = {"tag": args.tag, "commit": commit, "mesh": args.mesh}
        try:
            row.update(_run_cell(args, dm, pm, mesh, placements, N, D, variant, device))
            if row.get("ratios"):
                n_ok += 1
        except torch.cuda.OutOfMemoryError as e:
            row.update(
                {
                    "N": N,
                    "D": D,
                    "variant": variant,
                    "error": f"OOM: {e}"[:200],
                    "skip_reason": "oom",
                }
            )
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 -- one bad cell must not lose the rest of the sweep
            row.update(
                {"N": N, "D": D, "variant": variant, "error": f"{type(e).__name__}: {e}"[:400]}
            )
            _rank0_print(rank, traceback.format_exc())
        out_rows.append(row)
        _rank0_print(
            rank,
            "[CELL] "
            + json.dumps(
                {k: v for k, v in row.items() if not k.startswith("raw_max_")}, sort_keys=True
            ),
        )
        if args.out and rank == 0:
            with open(args.out, "w") as fh:
                json.dump(out_rows, fh, indent=1)
    done = len(out_rows)
    mark = "" if n_ok == len(cells) else "  INCOMPLETE"
    _rank0_print(rank, f"[DENOM] tag={args.tag} requested={len(cells)} ran={done} ok={n_ok}{mark}")
    return 0 if n_ok == len(cells) else 4


if __name__ == "__main__":
    sys.exit(main())
